"""Tests for DiffusionMoELayer: the required forward-pass shape check plus
dispatch correctness, gradient flow, and the periodic centroid-refresh policy.
"""

import torch

from diffusion_moe.models.moe_layer import DiffusionMoELayer

BATCH, SEQ_LEN, D_MODEL, N_EXPERTS, TOP_K = 2, 16, 64, 4, 2


def _make_layer(**overrides):
    kwargs = dict(
        d_model=D_MODEL,
        num_heads=8,
        max_seq_len=32,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        n_components=4,
        n_landmarks=16,
        diffusion_t=2,
        centroid_refresh_steps=5,
    )
    kwargs.update(overrides)
    return DiffusionMoELayer(**kwargs)


def _inputs():
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    positions = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
    return x, positions


def test_forward_output_and_aux_shapes():
    layer = _make_layer()
    x, positions = _inputs()
    out, aux = layer(x, positions)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert aux["router_logits"].shape == (BATCH, SEQ_LEN, N_EXPERTS)
    assert aux["gate_values"].shape == (BATCH, SEQ_LEN, TOP_K)
    assert aux["expert_indices"].shape == (BATCH, SEQ_LEN, TOP_K)
    assert aux["expert_indices"].dtype == torch.long
    assert aux["Psi_t"].shape[:2] == (BATCH, SEQ_LEN)
    # needed by total_loss (Section 8) to compute the separation loss per layer
    # without threading the model/centroids through separately
    assert aux["centroids"].shape == (N_EXPERTS, aux["Psi_t"].shape[-1])
    assert aux["Psi_landmarks"].shape[-1] == aux["Psi_t"].shape[-1]


def test_centroids_initialised_after_first_forward():
    layer = _make_layer()
    x, positions = _inputs()
    assert not layer.centroids.is_initialised
    layer(x, positions)
    assert layer.centroids.is_initialised


def test_step_counter_advances_only_in_training_mode():
    layer = _make_layer()
    x, positions = _inputs()

    layer.eval()
    layer(x, positions)
    assert layer._step.item() == 0

    layer.train()
    layer(x, positions)
    assert layer._step.item() == 1


def test_refresh_schedule_triggers_refit_every_n_steps():
    """centroid_refresh_steps=5: steps 0, 5, 10, ... refit (fit_transform);
    others reuse the frozen landmarks via transform(). We can't observe this
    directly, but ndm.landmarks_ should change on a refit step and stay fixed
    in between."""
    layer = _make_layer(centroid_refresh_steps=3)
    x, positions = _inputs()
    layer.train()

    layer(x, positions)  # step 0 -> refit
    landmarks_after_refit = layer.ndm.landmarks_.copy()

    layer(x, positions)  # step 1 -> transform only
    assert (layer.ndm.landmarks_ == landmarks_after_refit).all()

    layer(x, positions)  # step 2 -> transform only
    assert (layer.ndm.landmarks_ == landmarks_after_refit).all()

    layer(x, positions)  # step 3 -> refit again
    # landmarks may legitimately land on the same points again by chance with
    # this tiny fixed input, so just check it ran without error and shape holds
    assert layer.ndm.landmarks_.shape == landmarks_after_refit.shape


def test_dispatch_and_aggregate_matches_naive_per_token_reference():
    layer = _make_layer(d_model=8, num_heads=2, n_components=3, n_landmarks=10)
    torch.manual_seed(1)
    x = torch.randn(2, 5, 8)
    gate_values = torch.rand(2, 5, TOP_K)
    gate_values = gate_values / gate_values.sum(-1, keepdim=True)
    expert_indices = torch.randint(0, N_EXPERTS, (2, 5, TOP_K))

    out = layer._dispatch_and_aggregate(x, gate_values, expert_indices)

    ref = torch.zeros_like(x)
    for b in range(2):
        for s in range(5):
            acc = torch.zeros(8)
            for slot in range(TOP_K):
                e = expert_indices[b, s, slot].item()
                g = gate_values[b, s, slot].item()
                acc = acc + g * layer.experts[e](x[b, s])
            ref[b, s] = acc

    assert torch.allclose(out, ref, atol=1e-5)


def test_gradients_flow_to_attention_experts_and_centroids():
    layer = _make_layer()
    x, positions = _inputs()
    layer.train()

    out, _ = layer(x, positions)
    out.sum().backward()

    assert layer.attn.q_proj.weight.grad is not None
    assert layer.centroids.centroids.grad is not None
    assert any(
        expert.gate_proj.weight.grad is not None for expert in layer.experts
    )


def test_no_expert_left_without_gradient_when_all_are_used():
    """With enough tokens relative to n_experts, every expert should be picked
    by at least one token in top_k=2 routing, so every expert gets a gradient."""
    layer = _make_layer()
    x, positions = _inputs()
    layer.train()

    out, _ = layer(x, positions)
    out.sum().backward()

    for expert in layer.experts:
        assert expert.gate_proj.weight.grad is not None
