"""Learned expert centroids in diffusion coordinate space."""

from __future__ import annotations

import torch
from sklearn.cluster import kmeans_plusplus
from torch import nn


class ExpertCentroids(nn.Module):
    """Stores K centroids as an nn.Parameter of shape (K, n_components) in
    diffusion space. Centroids are seeded via k-means++ on the first batch of
    diffusion coordinates seen, then trained like any other parameter — pulled
    toward nearby tokens through the routing softmax's gradient, and pushed
    apart by centroid_separation_loss.
    """

    def __init__(self, n_experts: int, n_components: int, random_state: int = 42) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.n_components = n_components
        self.random_state = random_state

        self.centroids = nn.Parameter(torch.randn(n_experts, n_components) * 0.01)
        self.register_buffer("_initialised", torch.tensor(False))

    @property
    def is_initialised(self) -> bool:
        return bool(self._initialised)

    @torch.no_grad()
    def initialise_from_batch(self, Psi_t: torch.Tensor) -> None:
        """k-means++ seeding from a batch of diffusion coordinates. Psi_t: any
        shape (..., n_components); leading dims are flattened over tokens.
        A no-op after the first call (call once, on the first training batch).
        """
        if self.is_initialised:
            return

        flat = Psi_t.detach().reshape(-1, self.n_components).cpu().numpy()
        n_seeds = min(self.n_experts, flat.shape[0])

        centers, _ = kmeans_plusplus(
            flat, n_clusters=n_seeds, random_state=self.random_state
        )
        centroids = torch.from_numpy(centers).to(
            dtype=self.centroids.dtype, device=self.centroids.device
        )

        if n_seeds < self.n_experts:
            # Fewer distinct tokens than experts (e.g. a tiny batch) — jitter
            # copies of the last seed rather than leaving unseeded rows at zero.
            filler = centroids[-1:] + torch.randn(
                self.n_experts - n_seeds, self.n_components, device=centroids.device
            ) * 1e-3
            centroids = torch.cat([centroids, filler], dim=0)

        self.centroids.data.copy_(centroids)
        self._initialised.fill_(True)

    def forward(self) -> torch.Tensor:
        return self.centroids
