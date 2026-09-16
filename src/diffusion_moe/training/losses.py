"""Combined training loss: task cross-entropy + auxiliary routing losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusion_moe.data.dataset import IGNORE_INDEX
from diffusion_moe.routing.load_balance import coefficient_of_variation_loss
from diffusion_moe.routing.separation import centroid_separation_loss


def total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    router_outputs: dict[int, dict[str, torch.Tensor]],
    mu: float,
    nu: float,
) -> dict[str, torch.Tensor]:
    """Causal-LM cross-entropy plus the CV^2 load-balance and centroid-separation
    penalties, averaged over every MoE layer.

    logits: (batch, seq, vocab_size). labels: (batch, seq), shifted causal-LM
    targets with IGNORE_INDEX (-100) marking positions to skip — matches
    diffusion_moe.data.dataset.collate_fn's convention.

    router_outputs: {layer_idx: aux} for each MoE layer. Every variant's aux
    dict (DiffusionMoELayer, CosineMoELayer, SwitchMoELayer, RandomMoELayer)
    carries router_logits/gate_values/expert_indices, so load_loss is always
    computed uniformly across all of them. Only DiffusionMoELayer's aux also
    carries centroids/Psi_landmarks (its centroid-separation loss is specific
    to routing in diffusion coordinate space); sep_loss averages over just
    the layers that have both keys, and is 0 if none do. Empty router_outputs
    (a fully dense model) gives both losses 0.

    The load-balance loss uses softmax(router_logits) (untempered, ignoring
    the routing temperature tau) as the per-expert load distribution — a
    stable proxy for "how gating mass would spread absent hard top-k
    selection," in the same spirit as the Switch Transformer's auxiliary loss.

    mu, nu: weights for the load-balance and separation losses respectively.

    Returns a dict with "loss" (the weighted sum to call .backward() on),
    each aggregate component detached for logging, and a per-layer
    breakdown (`load_loss/layer_{idx}`, `sep_loss/layer_{idx}` for layers
    that have it) -- the aggregate mean hides exactly the thing worth
    watching when more than one MoE layer is active at once: whether
    collapse is uniform across layers or concentrated in a few (see
    phase1_findings_report.md §3.5/§3.6's "what happens if a layer keeps
    forcing a single expert to handle everything" discussion).
    """
    vocab_size = logits.shape[-1]
    task_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )

    per_layer: dict[str, torch.Tensor] = {}
    if router_outputs:
        load_losses = []
        sep_losses = []
        for layer_idx, aux in router_outputs.items():
            n_experts = aux["router_logits"].shape[-1]
            dense_weights = F.softmax(aux["router_logits"], dim=-1)
            layer_load_loss = coefficient_of_variation_loss(dense_weights, n_experts)
            load_losses.append(layer_load_loss)
            per_layer[f"load_loss/layer_{layer_idx}"] = layer_load_loss.detach()
            if "centroids" in aux and "Psi_landmarks" in aux:
                layer_sep_loss = centroid_separation_loss(aux["centroids"], aux["Psi_landmarks"])
                sep_losses.append(layer_sep_loss)
                per_layer[f"sep_loss/layer_{layer_idx}"] = layer_sep_loss.detach()
        load_loss = torch.stack(load_losses).mean()
        sep_loss = (
            torch.stack(sep_losses).mean()
            if sep_losses
            else torch.zeros((), device=logits.device, dtype=logits.dtype)
        )
    else:
        load_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        sep_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

    loss = task_loss + mu * load_loss + nu * sep_loss

    return {
        "loss": loss,
        "task_loss": task_loss.detach(),
        "load_loss": load_loss.detach(),
        "sep_loss": sep_loss.detach(),
        **per_layer,
    }
