"""Tests for RandomMoELayer: uniform random routing, no learned router params."""

import torch

from diffusion_moe.models.random_moe_layer import RandomMoELayer

BATCH, SEQ_LEN, D_MODEL, N_EXPERTS, TOP_K = 2, 16, 64, 4, 2


def _layer():
    return RandomMoELayer(
        d_model=D_MODEL, num_heads=8, max_seq_len=32, n_experts=N_EXPERTS, top_k=TOP_K
    )


def _inputs():
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    positions = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
    return x, positions


def test_shared_expert_toggleable_default_on():
    """use_shared_expert is a uniform toggle across all four router variants
    (default True) -- see DiffusionMoELayer's docstring for the motivation."""
    layer_on = RandomMoELayer(
        d_model=D_MODEL, num_heads=8, max_seq_len=32, n_experts=N_EXPERTS, top_k=TOP_K
    )
    assert layer_on.shared_expert is not None

    layer_off = RandomMoELayer(
        d_model=D_MODEL, num_heads=8, max_seq_len=32, n_experts=N_EXPERTS, top_k=TOP_K,
        use_shared_expert=False,
    )
    assert layer_off.shared_expert is None
    out, _ = layer_off(*_inputs())  # must not raise with shared_expert=None
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_output_and_aux_shapes():
    layer = _layer()
    x, positions = _inputs()
    out, aux = layer(x, positions)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert aux["router_logits"].shape == (BATCH, SEQ_LEN, N_EXPERTS)
    assert aux["gate_values"].shape == (BATCH, SEQ_LEN, TOP_K)
    assert aux["expert_indices"].shape == (BATCH, SEQ_LEN, TOP_K)
    assert aux["expert_indices"].dtype == torch.long


def test_no_learned_router_parameters():
    """The only learned parameters should belong to attention, norms, and
    experts — no separate router weight."""
    layer = _layer()
    param_names = [name for name, _ in layer.named_parameters()]
    assert not any("router" in name for name in param_names)


def test_gate_values_are_uniform_over_top_k():
    layer = _layer()
    x, positions = _inputs()
    _, aux = layer(x, positions)
    expected = torch.full_like(aux["gate_values"], 1.0 / TOP_K)
    assert torch.allclose(aux["gate_values"], expected)


def test_no_duplicate_expert_within_a_token():
    layer = _layer()
    x, positions = _inputs()
    _, aux = layer(x, positions)
    indices = aux["expert_indices"]
    for b in range(BATCH):
        for s in range(SEQ_LEN):
            row = indices[b, s].tolist()
            assert len(set(row)) == len(row)


def test_router_logits_uniform_softmax():
    layer = _layer()
    x, positions = _inputs()
    _, aux = layer(x, positions)
    assert torch.all(aux["router_logits"] == 0.0)


def test_routing_differs_across_calls():
    """Different forward calls should (almost certainly) produce different
    random assignments, unlike a deterministic router."""
    torch.manual_seed(0)
    layer = _layer()
    x, positions = _inputs()
    _, aux1 = layer(x, positions)
    _, aux2 = layer(x, positions)
    assert not torch.equal(aux1["expert_indices"], aux2["expert_indices"])


def test_gradients_flow_to_attention_and_experts():
    layer = _layer()
    x, positions = _inputs()
    out, _ = layer(x, positions)
    out.sum().backward()
    assert layer.attn.q_proj.weight.grad is not None
    assert any(e.gate_proj.weight.grad is not None for e in layer.experts)
