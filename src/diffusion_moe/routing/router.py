"""Diffusion-distance-based expert router."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiffusionRouter(nn.Module):
    """Routes tokens to experts by a softmax over negative squared diffusion
    distance to each expert's centroid, then selects the top-k experts per
    token.
    """

    def __init__(
        self, n_experts: int, top_k: int, tau: float = 0.1, noise_std: float = 0.0
    ) -> None:
        super().__init__()
        if top_k > n_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed n_experts ({n_experts})")
        self.n_experts = n_experts
        self.top_k = top_k
        self.tau = tau
        # Noisy top-k gating (Shazeer et al., 2017): i.i.d. Gaussian noise
        # added to the dense logits during training only, before top-k
        # selection and the tempered softmax. Targets a specific failure
        # mode found in a real pilot run: whichever expert centroids happen
        # to land closest to the bulk of the token distribution at k-means++
        # initialisation win an early lead in the router's tempered softmax,
        # and nothing in the clean (noise-free) formula gives the other
        # experts a chance to close that gap -- confirmed by a same-config,
        # different-seed rerun that reproduced comparably severe collapse
        # onto a *different* pair/triple of experts each time. Fixed
        # magnitude (not Shazeer's learned per-expert scale, which assumes a
        # linear router with its own weight matrix to house a second noise
        # projection -- this router's logits come from a fixed geometric
        # formula instead): meaningful as a fixed value specifically because
        # callers route on landmark_scale-normalised coordinates, so
        # router_logits is already in a consistent, dimensionless unit
        # regardless of layer or model (see routing/separation.py).
        self.noise_std = noise_std

    def forward(
        self, Psi_t: torch.Tensor, centroids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Psi_t: (batch, seq, n_components). centroids: (n_experts, n_components).

        Returns:
            gate_values: (batch, seq, top_k) — softmax weights of the selected
                experts, renormalised to sum to 1 over the top-k.
            expert_indices: (batch, seq, top_k) long — indices of the selected
                experts, in descending order of gate value.
            router_logits: (batch, seq, n_experts) — negative squared diffusion
                distance to every centroid, PLUS training-time noise if
                noise_std > 0 (dense, pre-top-k; used for the auxiliary
                load-balance / separation losses, so those losses see the
                same noisy signal top-k dispatch actually acted on).
        """
        # ||psi - c||^2 = ||psi||^2 - 2 psi . c + ||c||^2
        psi_sq = (Psi_t**2).sum(dim=-1, keepdim=True)  # (batch, seq, 1)
        c_sq = (centroids**2).sum(dim=-1)  # (n_experts,)
        cross = Psi_t @ centroids.t()  # (batch, seq, n_experts)
        sq_dist = psi_sq - 2 * cross + c_sq  # (batch, seq, n_experts)

        router_logits = -sq_dist
        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        weights = F.softmax(router_logits / self.tau, dim=-1)

        gate_values, expert_indices = self.top_k_gate(weights, self.top_k)
        return gate_values, expert_indices, router_logits

    @staticmethod
    def top_k_gate(weights: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Selects the top-k experts per token and renormalises their weights
        to sum to 1. weights: (..., n_experts). Returns (gate_values,
        expert_indices), each (..., k), sorted by descending weight.
        """
        top_values, top_indices = torch.topk(weights, k, dim=-1)
        gate_values = top_values / top_values.sum(dim=-1, keepdim=True)
        return gate_values, top_indices
