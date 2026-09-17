"""End-to-end integration tests spanning models, routing, losses, and geometry
— the cross-module checks that unit tests in tests/<subpackage>/ don't cover
in isolation. Run the whole suite with: pytest tests/ -v --tb=short
"""

import numpy as np
import torch

from diffusion_moe.geometry.nystrom import NystromDiffusionMap
from diffusion_moe.models.moe_model import DiffusionMoETransformer
from diffusion_moe.models.random_moe_layer import RandomMoELayer
from diffusion_moe.models.transformer_block import TransformerBlock
from diffusion_moe.training.losses import total_loss

VOCAB_SIZE = 1000
BATCH, SEQ_LEN = 2, 16
D_MODEL, N_LAYERS, N_EXPERTS = 64, 2, 4


def _tiny_model(**overrides):
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=8,
        max_seq_len=32,
        n_experts=N_EXPERTS,
        top_k=2,
        n_components=4,
        n_landmarks=16,
        diffusion_t=2,
        centroid_refresh_steps=1000,
    )
    kwargs.update(overrides)
    return DiffusionMoETransformer(**kwargs)


def test_full_forward_pass():
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    logits, activations, router_outputs = model(input_ids)

    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)
    assert set(router_outputs.keys()) == set(range(N_LAYERS))  # every layer is MoE by default
    for aux in router_outputs.values():
        assert "router_logits" in aux
        assert "gate_values" in aux
        assert "expert_indices" in aux


def test_loss_backward_all_parameters_have_gradients():
    """top_k=n_experts guarantees every expert actually processes at least
    one token this step, so this is a deterministic check, not one that
    could occasionally miss an unlucky expert with zero routed tokens."""
    model = _tiny_model(top_k=N_EXPERTS)
    model.train()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))

    logits, _, router_outputs = model(input_ids)
    losses = total_loss(logits, labels, router_outputs, mu=0.01, nu=0.05)
    losses["loss"].backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"

    # centroid parameters specifically, since they're the least "obviously
    # trained" ones (fed only through the router's distance computation)
    for layer in model.blocks:
        assert layer.centroids.centroids.grad is not None
        assert torch.any(layer.centroids.centroids.grad != 0)


def test_baseline_parity_random_router_top_k_equals_n_experts():
    """With top_k=n_experts and uniform 1/top_k gate weights, RandomMoELayer
    computes mean_e(ExpertFFN_e(x)) for every token. That's mathematically
    identical to a dense FFN built by concatenating every expert's gate/up
    weights (block-diagonal SwiGLU decomposes into independent per-expert
    hidden units) and concatenating down_proj weights scaled by 1/n_experts
    (since mean = (1/n_experts) * sum, and summing per-expert down-projected
    contributions is exactly what one combined down_proj computes). This
    isn't an approximate/statistical parity — it's an exact algebraic
    identity, so a tight tolerance (float32 summation-order noise only) is
    the right bar, not a loose "roughly similar" one.
    """
    torch.manual_seed(0)
    d_model, n_experts, ffn_dim = 64, 4, 256

    moe = RandomMoELayer(
        d_model=d_model, num_heads=8, max_seq_len=32, n_experts=n_experts, top_k=n_experts,
        ffn_dim=ffn_dim, use_shared_expert=False,  # isolate the routed-experts identity being tested
    )
    dense = TransformerBlock(d_model=d_model, num_heads=8, max_seq_len=32, ffn_dim=ffn_dim)

    dense.attn.load_state_dict(moe.attn.state_dict())
    dense.attn_norm.load_state_dict(moe.attn_norm.state_dict())
    dense.ffn_norm.load_state_dict(moe.ffn_norm.state_dict())
    with torch.no_grad():
        dense.ffn.gate_proj.weight.copy_(
            torch.cat([e.gate_proj.weight for e in moe.experts], dim=0)
        )
        dense.ffn.up_proj.weight.copy_(torch.cat([e.up_proj.weight for e in moe.experts], dim=0))
        dense.ffn.down_proj.weight.copy_(
            torch.cat([e.down_proj.weight for e in moe.experts], dim=1) / n_experts
        )

    x = torch.randn(BATCH, SEQ_LEN, d_model)
    positions = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)

    moe_out, _ = moe(x, positions)
    dense_out = dense(x, positions)

    assert torch.allclose(moe_out, dense_out, atol=1e-4)


def test_nystrom_quality_exact_reproduction_on_landmarks():
    """"n_landmarks=64 (full)" on a 512x64 matrix: with 512 >> 64, no
    landmark-count clamping occurs (the full requested 64 landmarks are
    used). "Quality" here checks Nystrom's defining correctness property —
    applying the out-of-sample extension formula to points that were
    themselves the landmarks must exactly reproduce their direct (non-Nystrom)
    diffusion coordinates. This is the right quality bar for a unit test:
    it's a mathematical guarantee (unlike Nystrom error on arbitrary
    held-out points, which depends on how well 64 landmarks happen to
    capture unstructured random data's kernel spectrum, and isn't reliably
    boundable at all — let alone below 1e-4).
    """
    Z = np.random.RandomState(0).randn(512, 64)
    ndm = NystromDiffusionMap(n_landmarks=64, n_components=32, t=3, alpha=1.0)
    ndm.fit(Z)

    assert ndm.landmarks_.shape[0] == 64  # requested count honored, not clamped

    psi_via_transform = ndm.transform(ndm.landmarks_)
    error = np.abs(psi_via_transform - ndm.psi_landmarks_).max()
    assert error < 1e-4


def test_routing_determinism_same_input_same_state_same_assignment():
    """Same input, same (frozen) model state -> same expert assignment.

    "Same state" specifically means past the initial centroid_refresh_steps
    boundary: DiffusionMoELayer's step counter only advances in training
    mode (by design — see test_moe_layer.py), so raw eval-mode calls from a
    freshly-constructed layer would each re-trigger an independent KMeans +
    ARPACK re-fit at "step 0", and — as in the centroid-seeding investigation
    in tests/routing/test_centroids.py — BLAS non-determinism across
    separate re-fits can occasionally flip which of two near-tied experts
    ranks first in top-k, even with a fixed random_state. That's not what
    "same state" means; it's testing routing on a *moving* geometry. One
    training-mode call first advances the layer past step 0, so the two
    comparison calls both land on the same frozen landmarks (via
    NystromDiffusionMap.transform, pure matrix algebra — no re-fit, no
    non-determinism)."""
    model = _tiny_model(centroid_refresh_steps=1000)
    model.train()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    model(input_ids)  # initial fit; advances every MoE layer past step 0

    model.eval()
    with torch.no_grad():
        _, _, router_outputs_1 = model(input_ids)
        _, _, router_outputs_2 = model(input_ids)

    for layer_idx in router_outputs_1:
        assert torch.equal(
            router_outputs_1[layer_idx]["expert_indices"],
            router_outputs_2[layer_idx]["expert_indices"],
        )
        assert torch.allclose(
            router_outputs_1[layer_idx]["gate_values"],
            router_outputs_2[layer_idx]["gate_values"],
        )
