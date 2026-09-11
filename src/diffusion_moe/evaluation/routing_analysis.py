"""Routing diagnostics: per-token expert assignment, gate values, and
diffusion coordinates, collected by hooking every DiffusionMoELayer in a
DiffusionMoETransformer."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch

from diffusion_moe.models.moe_layer import DiffusionMoELayer


class RoutingAnalyser:
    """Hooks into every DiffusionMoELayer of a DiffusionMoETransformer and
    records, on each forward pass, every token's routed expert(s), gate
    value(s), and diffusion coordinates — keyed by layer index.

    Usage:
        analyser = RoutingAnalyser(model)
        with analyser:
            for batch in eval_loader:
                model(batch["input_ids"])
        counts = analyser.expert_token_counts()

    Kept attached across many calls (e.g. logged periodically through a
    training run rather than a single eval pass), it also accumulates a
    per-call centroid snapshot per layer, for centroid_distances_over_training.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.moe_layers: dict[int, DiffusionMoELayer] = {
            idx: module
            for idx, module in enumerate(getattr(model, "blocks", []))
            if isinstance(module, DiffusionMoELayer)
        }
        if not self.moe_layers:
            raise ValueError(
                "No DiffusionMoELayer found in model.blocks — RoutingAnalyser "
                "has nothing to hook into."
            )
        self.n_experts = next(iter(self.moe_layers.values())).n_experts

        self.records: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(list)
        self._centroid_snapshots: dict[int, list[torch.Tensor]] = defaultdict(list)
        self._handles: list[Any] = []

    def _make_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            _, aux = output
            self.records[layer_idx].append(
                {
                    "expert_indices": aux["expert_indices"].detach().cpu(),
                    "gate_values": aux["gate_values"].detach().cpu(),
                    "Psi_t": aux["Psi_t"].detach().cpu(),
                }
            )
            self._centroid_snapshots[layer_idx].append(aux["centroids"].detach().cpu().clone())

        return hook

    def __enter__(self) -> "RoutingAnalyser":
        self.records = defaultdict(list)
        self._centroid_snapshots = defaultdict(list)
        self._handles = [
            layer.register_forward_hook(self._make_hook(idx))
            for idx, layer in self.moe_layers.items()
        ]
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def expert_token_counts(self, layer_idx: int | None = None) -> dict[int, int]:
        """{expert_id: number of (token, slot) assignments} across every
        recorded forward pass. Aggregates over all MoE layers if layer_idx is
        None, otherwise restricts to that one layer."""
        counts = {e: 0 for e in range(self.n_experts)}
        layer_indices = [layer_idx] if layer_idx is not None else list(self.records)
        for idx in layer_indices:
            for record in self.records[idx]:
                flat = record["expert_indices"].reshape(-1)
                values, value_counts = torch.unique(flat, return_counts=True)
                for expert_id, count in zip(values, value_counts):
                    counts[int(expert_id)] += int(count)
        return counts

    def routing_entropy_per_expert(self, layer_idx: int | None = None) -> dict[int, float]:
        """Each expert's contribution to the Shannon entropy (in bits) of the
        empirical expert-assignment distribution: -p_e * log2(p_e), where p_e
        is the fraction of (token, slot) assignments routed to expert e.

        Summing over all experts gives the aggregate routing entropy (maximum
        log2(n_experts) at a perfectly uniform load); an expert whose
        contribution has collapsed toward 0 is getting starved of traffic.
        """
        counts = self.expert_token_counts(layer_idx)
        total = sum(counts.values())
        entropy_per_expert: dict[int, float] = {}
        for expert_id, count in counts.items():
            if total == 0 or count == 0:
                entropy_per_expert[expert_id] = 0.0
                continue
            p = count / total
            entropy_per_expert[expert_id] = -p * math.log2(p)
        return entropy_per_expert

    def centroid_distances_over_training(self, layer_idx: int) -> list[float]:
        """Mean pairwise centroid distance at each recorded snapshot for a
        given layer — tracks whether experts are drifting apart (learning
        distinct specialisations) or collapsing together over time."""
        distances = []
        for centroids in self._centroid_snapshots[layer_idx]:
            n = centroids.shape[0]
            pairwise = torch.cdist(centroids, centroids, p=2)
            mask = ~torch.eye(n, dtype=torch.bool)
            distances.append(float(pairwise[mask].mean()))
        return distances
