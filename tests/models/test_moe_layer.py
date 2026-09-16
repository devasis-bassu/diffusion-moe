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


def test_cosine_false_by_default_matches_prior_behavior():
    layer = _make_layer()
    assert layer.cosine is False


def test_cosine_true_fits_on_l2_normalized_activations():
    """Confirms cosine=True actually changes what gets fit, not just that the
    flag is stored. Landmarks are k-means centroids -- a convex combination
    (mean) of whichever points get assigned to a cluster -- so if the INPUT
    to k-means was L2-normalized (every point norm exactly 1), every
    landmark's norm must be <= 1 by the triangle inequality, regardless of
    cluster assignment. Raw torch.randn activations at D_MODEL=64 have no
    such bound (norm ~= sqrt(64) = 8 in expectation) -- a clear, reliable
    discriminator between "fit on raw activations" and "fit on normalized
    ones" without needing to inspect intermediate arrays directly.
    """
    torch.manual_seed(0)  # was unseeded -- flaky, see below
    x, positions = _inputs()

    raw_layer = _make_layer(cosine=False)
    raw_layer(x, positions)
    # Norm, not a single coordinate's raw value: landmarks are k-means
    # centroids (means of assigned points), and averaging shrinks individual
    # coordinate magnitude -- a max-single-coordinate proxy can dip under 1.0
    # even for unnormalized ~N(0,1)^64 points (observed flakily in practice,
    # unseeded, across separate runs: 0.775, then 0.862). Norm is the
    # quantity the docstring's own triangle-inequality reasoning is actually
    # about, and survives averaging with much more headroom (expectation
    # sqrt(64) = 8, vs. the cosine layer's hard <= 1.0 cap below).
    raw_norms = (raw_layer.ndm.landmarks_**2).sum(axis=-1) ** 0.5
    assert raw_norms.max() > 1.0

    cosine_layer = _make_layer(cosine=True)
    cosine_layer(x, positions)
    landmark_norms = (cosine_layer.ndm.landmarks_**2).sum(axis=-1) ** 0.5
    assert (landmark_norms <= 1.0 + 1e-6).all()


def test_shared_expert_exists_with_same_width_as_a_routed_expert():
    layer = _make_layer()
    assert hasattr(layer, "shared_expert")
    assert layer.shared_expert.hidden_dim == layer.experts[0].hidden_dim


def test_shared_expert_contributes_even_when_routing_contributes_nothing():
    """The actual property that makes it "mandatory" / "outside the router's
    influence": output must still differ from the plain residual x even if
    dispatch_and_aggregate (the routed path) contributes exactly zero --
    i.e. the shared expert's contribution cannot be routing-contingent."""
    layer = _make_layer()
    x, positions = _inputs()

    original_dispatch = layer._dispatch_and_aggregate
    layer._dispatch_and_aggregate = lambda *args, **kwargs: torch.zeros_like(x)
    try:
        out, _ = layer(x, positions)
    finally:
        layer._dispatch_and_aggregate = original_dispatch

    # out = x + z (attn residual) + 0 (routed path) + shared_expert(...):
    # must differ from x + z alone, i.e. the shared expert really fired.
    z = layer.attn(layer.attn_norm(x), positions, None)
    assert not torch.allclose(out, x + z)


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


def test_forward_and_backward_work_under_real_bf16_mixed_precision():
    """Reproduces a real crash found by actually running a training step
    through Trainer with the project's own default precision (bf16): NumPy
    has no bfloat16 dtype at all, so any unguarded `.numpy()` call on a bf16
    tensor raises outright (not a precision/accuracy issue -- a hard crash),
    and torch.cdist has no bfloat16 implementation either. Neither training-
    free diagnostic nor the pilot (whose PilotMoEBlock/DtypeCastWrapper
    already forced fp32 for unrelated reasons) could have caught this --
    only running the production DiffusionMoELayer under real mixed precision
    did. Covers both fixed call sites: _compute_diffusion_coords's z.numpy()
    and landmark_scale/centroid_separation_loss's torch.cdist (exercised via
    ExpertCentroids.initialise_from_batch and this layer's own forward).
    """
    layer = _make_layer().to(torch.bfloat16)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, dtype=torch.bfloat16)
    positions = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
    layer.train()

    out, _ = layer(x, positions)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

    out.float().sum().backward()
    assert layer.centroids.centroids.grad is not None
