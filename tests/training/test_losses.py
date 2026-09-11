"""Tests for total_loss: task/load/separation combination and edge cases."""

import torch
import torch.nn.functional as F

from diffusion_moe.data.dataset import IGNORE_INDEX
from diffusion_moe.routing.load_balance import coefficient_of_variation_loss
from diffusion_moe.routing.separation import centroid_separation_loss
from diffusion_moe.training.losses import total_loss

BATCH, SEQ_LEN, VOCAB_SIZE = 2, 8, 50
N_EXPERTS, N_COMPONENTS, N_LANDMARKS = 4, 6, 20


def _logits_and_labels(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE, generator=g)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), generator=g)
    return logits, labels


def _fake_router_outputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    router_logits = torch.randn(BATCH, SEQ_LEN, N_EXPERTS, generator=g)
    centroids = torch.randn(N_EXPERTS, N_COMPONENTS, generator=g)
    psi_landmarks = torch.randn(N_LANDMARKS, N_COMPONENTS, generator=g)
    return {
        0: {
            "router_logits": router_logits,
            "centroids": centroids,
            "Psi_landmarks": psi_landmarks,
        }
    }


def test_empty_router_outputs_gives_zero_aux_losses():
    logits, labels = _logits_and_labels()
    result = total_loss(logits, labels, router_outputs={}, mu=0.5, nu=0.5)

    assert torch.isclose(result["load_loss"], torch.tensor(0.0))
    assert torch.isclose(result["sep_loss"], torch.tensor(0.0))
    expected_task = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    assert torch.isclose(result["loss"], expected_task, atol=1e-5)
    assert torch.isclose(result["task_loss"], expected_task, atol=1e-5)


def test_ignore_index_excludes_padded_positions():
    logits, labels = _logits_and_labels()
    labels_with_padding = labels.clone()
    labels_with_padding[:, -1] = IGNORE_INDEX

    result_full = total_loss(logits, labels, router_outputs={}, mu=0.0, nu=0.0)
    result_padded = total_loss(logits, labels_with_padding, router_outputs={}, mu=0.0, nu=0.0)

    assert not torch.isclose(result_full["task_loss"], result_padded["task_loss"])


def test_matches_manual_component_computation():
    logits, labels = _logits_and_labels()
    router_outputs = _fake_router_outputs()
    mu, nu = 0.1, 0.2

    result = total_loss(logits, labels, router_outputs, mu=mu, nu=nu)

    aux = router_outputs[0]
    expected_load = coefficient_of_variation_loss(
        F.softmax(aux["router_logits"], dim=-1), N_EXPERTS
    )
    expected_sep = centroid_separation_loss(aux["centroids"], aux["Psi_landmarks"])
    expected_task = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    expected_loss = expected_task + mu * expected_load + nu * expected_sep

    assert torch.isclose(result["load_loss"], expected_load, atol=1e-5)
    assert torch.isclose(result["sep_loss"], expected_sep, atol=1e-5)
    assert torch.isclose(result["loss"], expected_loss, atol=1e-5)


def test_averages_across_multiple_moe_layers():
    router_outputs = _fake_router_outputs(seed=0)
    router_outputs[1] = _fake_router_outputs(seed=1)[0]
    logits, labels = _logits_and_labels()

    result = total_loss(logits, labels, router_outputs, mu=1.0, nu=1.0)

    manual_loads = [
        coefficient_of_variation_loss(F.softmax(aux["router_logits"], dim=-1), N_EXPERTS)
        for aux in router_outputs.values()
    ]
    expected_load = torch.stack(manual_loads).mean()
    assert torch.isclose(result["load_loss"], expected_load, atol=1e-5)


def test_component_tensors_are_detached_but_loss_has_grad():
    logits, labels = _logits_and_labels()
    logits = logits.clone().requires_grad_(True)
    router_outputs = _fake_router_outputs()
    # make centroids require grad to exercise the detach behaviour
    router_outputs[0]["centroids"] = router_outputs[0]["centroids"].requires_grad_(True)

    result = total_loss(logits, labels, router_outputs, mu=0.1, nu=0.1)

    assert result["loss"].requires_grad
    assert not result["task_loss"].requires_grad
    assert not result["load_loss"].requires_grad
    assert not result["sep_loss"].requires_grad

    result["loss"].backward()
    assert logits.grad is not None


def test_mu_scales_load_loss_contribution():
    logits, labels = _logits_and_labels()
    router_outputs = _fake_router_outputs()

    low_mu = total_loss(logits, labels, router_outputs, mu=0.0, nu=0.0)
    high_mu = total_loss(logits, labels, router_outputs, mu=10.0, nu=0.0)

    # same task/aux components, only the weighted total should differ (unless
    # load_loss happens to be exactly zero, which fake random logits won't be)
    assert not torch.isclose(low_mu["loss"], high_mu["loss"])
    assert torch.isclose(low_mu["load_loss"], high_mu["load_loss"], atol=1e-5)
