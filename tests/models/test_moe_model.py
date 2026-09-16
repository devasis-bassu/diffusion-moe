"""Tests for DiffusionMoETransformer: the required forward-pass shape check,
plus layers_to_replace ablation wiring."""

import pytest
import torch

from diffusion_moe.models.cosine_moe_layer import CosineMoELayer
from diffusion_moe.models.moe_layer import DiffusionMoELayer
from diffusion_moe.models.moe_model import DiffusionMoETransformer
from diffusion_moe.models.random_moe_layer import RandomMoELayer
from diffusion_moe.models.switch_moe_layer import SwitchMoELayer
from diffusion_moe.models.transformer_block import TransformerBlock

BATCH, SEQ_LEN, D_MODEL, N_LAYERS = 2, 16, 64, 3
N_EXPERTS, TOP_K = 4, 2
VOCAB_SIZE = 1000


def _model(**overrides):
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=8,
        max_seq_len=32,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        n_components=4,
        n_landmarks=16,
        diffusion_t=2,
        centroid_refresh_steps=5,
    )
    kwargs.update(overrides)
    return DiffusionMoETransformer(**kwargs)


def test_token_embedding_uses_small_std_not_default_normal_1():
    """nn.Embedding's default init is Normal(0, 1) -- std=1.0. Dramatically
    too large for a transformer embedding table, and especially costly since
    tie_embeddings=True by default makes this same matrix double as the
    LM-head unembedding projection. Root-caused a real anomaly: this
    project's own from-scratch 24-layer/every-layer-MoE config had initial
    task_loss ~823 nats against a random-guessing baseline of
    ln(32000)=10.4 -- traced to logits with max abs ~828 at random init,
    which in turn traced to this embedding init."""
    torch.manual_seed(0)
    model = _model()
    assert model.token_embedding.weight.std().item() < 0.1


def test_residual_output_projections_are_scaled_down_by_depth():
    """The other half of the same fix: out_proj (attention) and down_proj
    (every FFN -- dense, routed expert, and shared expert alike) should be
    visibly smaller than what nn.Linear's own default init alone would give,
    scaled by 1/sqrt(2*n_layers) (GPT-2/nanoGPT convention) -- otherwise
    residual-stream variance compounds with depth."""
    torch.manual_seed(0)
    scaled_model = _model()

    torch.manual_seed(0)
    unscaled_out_proj = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)  # same shape as out_proj

    scale = 1.0 / (2 * N_LAYERS) ** 0.5
    expected_std = unscaled_out_proj.weight.std().item() * scale
    actual_std = scaled_model.blocks[0].attn.out_proj.weight.std().item()
    assert abs(actual_std - expected_std) / expected_std < 0.15  # same seed, same shape -> close


def test_random_init_task_loss_near_random_guessing_baseline():
    """The actual end-to-end check: cross-entropy of a freshly initialized
    model against random labels should land near ln(vocab_size) -- a
    properly-calibrated random init is, by construction, close to a uniform
    guess over the vocabulary. Before the embedding-std and residual-output-
    scaling fixes, this was ~80x too high on the real 300M/24-layer config
    (823 nats vs. ln(32000)=10.4); this test uses a tiny config for speed,
    but checks the same invariant."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    model = _model()
    model.eval()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    with torch.no_grad():
        logits, _, _ = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))

    import math

    random_baseline = math.log(VOCAB_SIZE)
    assert loss.item() < random_baseline * 2  # generous margin, still catches an 80x blowup


def test_forward_pass_shapes_batch2_seq16_dmodel64_experts4_topk2():
    model = _model()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits, activations, router_outputs = model(input_ids)

    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)
    assert set(activations.keys()) == set(range(N_LAYERS))
    for act in activations.values():
        assert act.shape == (BATCH, SEQ_LEN, D_MODEL)

    # default layers_to_replace=None -> every layer is a DiffusionMoELayer
    assert set(router_outputs.keys()) == set(range(N_LAYERS))
    for aux in router_outputs.values():
        assert aux["gate_values"].shape == (BATCH, SEQ_LEN, TOP_K)
        assert aux["expert_indices"].shape == (BATCH, SEQ_LEN, TOP_K)


def test_all_layers_are_moe_by_default():
    model = _model()
    assert all(isinstance(block, DiffusionMoELayer) for block in model.blocks)


def test_layers_to_replace_subset_mixes_dense_and_moe_blocks():
    model = _model(layers_to_replace=[1])
    assert isinstance(model.blocks[0], TransformerBlock)
    assert isinstance(model.blocks[1], DiffusionMoELayer)
    assert isinstance(model.blocks[2], TransformerBlock)

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits, activations, router_outputs = model(input_ids)

    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)
    assert set(router_outputs.keys()) == {1}
    assert set(activations.keys()) == set(range(N_LAYERS))


def test_layers_to_replace_empty_list_is_fully_dense():
    model = _model(layers_to_replace=[])
    assert all(isinstance(block, TransformerBlock) for block in model.blocks)

    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits, _, router_outputs = model(input_ids)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)
    assert router_outputs == {}


def test_invalid_layer_index_raises():
    with pytest.raises(ValueError):
        _model(layers_to_replace=[N_LAYERS])  # out of range


def test_cosine_layers_sets_cosine_flag_only_on_the_given_diffusion_layers():
    """cosine_layers is per-layer, not a project-wide switch (see the
    recommendation this implements: phase1_findings_report.md §2.4/§7's
    recommendation 2 -- cosine is a fix for specific layers, not a uniform
    default). Layer 2 should get cosine=True, layers 0 and 1 (also
    DiffusionMoELayer, since layers_to_replace defaults to every layer)
    should not."""
    model = _model(cosine_layers=[2])
    assert model.blocks[0].cosine is False
    assert model.blocks[1].cosine is False
    assert model.blocks[2].cosine is True


def test_cosine_layers_index_outside_layers_to_replace_is_inert():
    """A cosine_layers index for a layer that isn't itself replaced (still
    dense) has nothing to apply to -- must not raise, and the dense block
    stays a plain TransformerBlock."""
    model = _model(layers_to_replace=[1], cosine_layers=[0, 1])
    assert isinstance(model.blocks[0], TransformerBlock)
    assert model.blocks[1].cosine is True


def test_cosine_layers_defaults_to_no_layers_using_cosine():
    model = _model()
    assert all(block.cosine is False for block in model.blocks)


def test_invalid_cosine_layers_index_raises():
    with pytest.raises(ValueError):
        _model(cosine_layers=[N_LAYERS])  # out of range


def test_noise_std_threaded_to_every_diffusion_layers_router():
    model = _model(noise_std=0.25)
    for block in model.blocks:
        assert block.router.noise_std == 0.25


@pytest.mark.parametrize(
    "router,expected_class",
    [
        ("diffusion", DiffusionMoELayer),
        ("cosine", CosineMoELayer),
        ("switch", SwitchMoELayer),
        ("random", RandomMoELayer),
    ],
)
def test_router_field_selects_correct_layer_class(router, expected_class):
    model = _model(router=router)
    assert all(isinstance(block, expected_class) for block in model.blocks)


@pytest.mark.parametrize("router", ["cosine", "switch", "random"])
def test_baseline_router_forward_pass_shapes(router):
    model = _model(router=router)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits, activations, router_outputs = model(input_ids)

    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)
    assert set(router_outputs.keys()) == set(range(N_LAYERS))
    for aux in router_outputs.values():
        assert aux["gate_values"].shape == (BATCH, SEQ_LEN, TOP_K)


def test_unknown_router_raises_value_error():
    with pytest.raises(ValueError):
        _model(router="not_a_real_router")


def test_mixed_layers_to_replace_with_baseline_router():
    model = _model(router="switch", layers_to_replace=[1])
    assert isinstance(model.blocks[0], TransformerBlock)
    assert isinstance(model.blocks[1], SwitchMoELayer)
    assert isinstance(model.blocks[2], TransformerBlock)


def test_weight_tying_enabled_by_default():
    model = _model()
    assert model.lm_head.weight is model.token_embedding.weight


def test_gradients_flow_through_full_model():
    model = _model(layers_to_replace=[0, 2])
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    model.train()

    logits, _, _ = model(input_ids)
    logits.sum().backward()

    assert model.token_embedding.weight.grad is not None
    moe_layer = model.blocks[0]
    assert moe_layer.centroids.centroids.grad is not None
