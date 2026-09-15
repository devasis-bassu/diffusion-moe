"""Tests for PilotMoEBlock: the drop-in `.mlp` replacement used by the
bounded pilot fine-tune (scripts/pilot_finetune.py). Mirrors
test_moe_layer.py's checks, adapted for PilotMoEBlock's simpler forward(x)
-> Tensor signature (no attention of its own) and its last_aux side-channel
(the surrounding frozen decoder layer only expects a plain tensor back, so
routing info can't be returned directly the way DiffusionMoELayer does).
"""

import torch
import torch.nn.functional as F

from diffusion_moe.models.pilot_moe_block import PilotMoEBlock
from diffusion_moe.routing.separation import landmark_scale
from diffusion_moe.training.losses import total_loss

BATCH, SEQ_LEN, D_MODEL, FFN_DIM, N_EXPERTS, TOP_K = 2, 16, 64, 128, 4, 2


def _make_block(**overrides):
    kwargs = dict(
        d_model=D_MODEL,
        ffn_dim=FFN_DIM,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        n_components=4,
        n_landmarks=16,
        diffusion_t=2,
        centroid_refresh_steps=3,
    )
    kwargs.update(overrides)
    return PilotMoEBlock(**kwargs)


def test_forward_output_shape_and_aux():
    block = _make_block()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = block(x)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    aux = block.last_aux
    assert aux["router_logits"].shape == (BATCH, SEQ_LEN, N_EXPERTS)
    assert aux["gate_values"].shape == (BATCH, SEQ_LEN, TOP_K)
    assert aux["expert_indices"].shape == (BATCH, SEQ_LEN, TOP_K)
    assert aux["Psi_t"].shape[:2] == (BATCH, SEQ_LEN)
    assert aux["centroids"].shape == (N_EXPERTS, aux["Psi_t"].shape[-1])
    assert aux["Psi_landmarks"].shape[-1] == aux["Psi_t"].shape[-1]


def test_centroids_initialised_after_first_forward():
    block = _make_block()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    assert not block.centroids.is_initialised
    block(x)
    assert block.centroids.is_initialised


def test_step_counter_advances_only_in_training_mode():
    block = _make_block()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    block.eval()
    block(x)
    assert block._step.item() == 0

    block.train()
    block(x)
    assert block._step.item() == 1


def test_refresh_schedule_reuses_landmarks_between_refits():
    block = _make_block(centroid_refresh_steps=3)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    block.train()

    block(x)  # step 0 -> refit
    landmarks_after_refit = block.ndm.landmarks_.copy()

    block(x)  # step 1 -> transform only
    assert (block.ndm.landmarks_ == landmarks_after_refit).all()

    block(x)  # step 2 -> transform only
    assert (block.ndm.landmarks_ == landmarks_after_refit).all()


def test_gradients_flow_to_experts_and_centroids():
    block = _make_block()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    block.train()

    out = block(x)
    out.sum().backward()

    assert block.centroids.centroids.grad is not None
    assert any(expert.gate_proj.weight.grad is not None for expert in block.experts)
    assert x.grad is not None  # gradient must flow back into the frozen backbone


def test_last_aux_plugs_directly_into_total_loss():
    """The whole point of last_aux: a single PilotMoEBlock's aux dict should
    slot straight into total_loss's router_outputs convention, the same as
    any of the project's own MoE layer variants.
    """
    block = _make_block()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    block.train()
    block(x)

    vocab_size = 50
    logits = torch.randn(BATCH, SEQ_LEN, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (BATCH, SEQ_LEN))

    result = total_loss(logits, labels, {2: block.last_aux}, mu=0.01, nu=0.05)

    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert logits.grad is not None


def test_scale_normalization_prevents_uniform_dispatch_from_tiny_diffusion_coordinates():
    """Reproduces the real-world failure mode found via a live training run
    on real Mistral-7B activations: diffusion coordinates
    (Psi_t = eigenvector * eigenvalue^t) can be vanishingly small in
    absolute magnitude — forced here with a large diffusion_t, the same
    mechanism that made it happen for real (small eigenvalues raised to a
    power) — which, without scale normalization, would make tau=0.1 give a
    dispatch softmax numerically indistinguishable from uniform regardless
    of which centroid is actually closest. That's exactly what the real
    training run showed before this fix: every expert reading exactly
    1/n_experts on every step.
    """
    block = _make_block(diffusion_t=20, tau=0.1)
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    block.train()

    block(x)
    aux = block.last_aux

    # Sanity check this test actually exercises the failure mode: the RAW
    # (pre-fix) landmark scale should indeed be forced tiny by diffusion_t=20.
    raw_scale = landmark_scale(torch.from_numpy(block.ndm.psi_landmarks_).float())
    assert raw_scale < 1e-3

    # aux carries the POST-fix, rescaled coordinates -> should be back to a
    # sane, O(1) scale regardless of how tiny the raw ones were.
    assert torch.isclose(landmark_scale(aux["Psi_landmarks"]), torch.tensor(1.0), atol=0.5)

    dense_weights = F.softmax(aux["router_logits"] / block.router.tau, dim=-1)
    per_expert_load = dense_weights.reshape(-1, N_EXPERTS).mean(dim=0)
    # Real spread across experts, NOT the degenerate uniform 1/n_experts
    # dispatch the unfixed version produces from coordinates this tiny.
    assert per_expert_load.std().item() > 1e-4
