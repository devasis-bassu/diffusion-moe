"""Tests whether the FFN-width-reduction hypothesis actually holds, as
opposed to only the routing-manifold hypothesis Phase 1's geometry kill
switch checks.

A low intrinsic dimension r* on the diffusion coordinates says a coarse,
low-frequency manifold organizes *routing* — it says nothing about whether
each region's FFN computation is correspondingly simpler. The diffusion map
is an explicit low-pass filter (Psi_t downweights each mode by lambda^t), so
a small r* is exactly consistent with there being plenty of high-frequency
structure left over that two tokens in the same coarse cluster still differ
on — and the project's own framing of the FFN's job (Step 4: attention
blurs, the FFN sharpens) is precisely about resolving that high-frequency
detail. expert_ffn.py's width formula (d_ff^(k) ~= d_ff / K) and the G2
success criterion both assume, additionally and separately from r*, that a
narrower per-cluster FFN retains enough capacity — that's an unverified
leap the geometry module alone can't detect.

This module tests that leap directly using the PRETRAINED DENSE model's own
FFN (Geva et al.: rows of the up-projection act as pattern detectors — a
neuron "fires" when the token resembles its learned prototype). For a
candidate cluster count K (e.g. matching configs/base_config.yaml's
n_experts), it clusters tokens by their diffusion coordinates (a stand-in
for how DiffusionRouter would route them) and asks: does each cluster rely
on a small, largely DISJOINT subset of the d_ff neurons the way narrow
per-expert FFNs would need it to, or does every cluster need roughly the
same broad set of neurons regardless of which coarse region it's in? The
latter would falsify the width-reduction hypothesis independent of r*,
without training anything.

CAVEAT this module cannot resolve on its own: the dense model was never
trained with any incentive to organize its FFN around diffusion-cluster
boundaries, so a weak result against the diffusion partition specifically
under-determines whether specialization is achievable at all versus just
not found by this particular (untrained, fixed) router signal — training a
real MoE actively reshapes specialization via its load-balancing loss,
which nothing here can simulate. oracle_cluster_tokens_by_activation and
cluster_agreement below split that ambiguity into two separately-answerable
pieces without training anything: (1) cluster tokens directly by their OWN
FFN activation pattern — the best-case partition for this exact metric,
establishing a ceiling on achievable specialization independent of any
routing signal; (2) measure how much of that oracle partition's structure
the diffusion clustering already recovers. Weak specialization against the
oracle ceiling means the FFN's computation isn't cleanly separable by ANY
grouping — real evidence against width reduction. Strong oracle
specialization with low diffusion-vs-oracle agreement means specialization
is achievable in principle, just not by this particular fixed signal — a
router/training problem, not a fundamental impossibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class MLPInputCapture:
    """Registers forward PRE-hooks on every decoder layer's `mlp.down_proj`,
    capturing its input — which for a SwiGLU MLP
    (down_proj(act_fn(gate_proj(x)) * up_proj(x))) is exactly the full
    d_ff-wide gated intermediate activation, i.e. the same object Geva et
    al.'s "pattern detector" analysis operates on. Hooking down_proj's INPUT
    rather than re-deriving gate_proj/up_proj/act_fn by hand keeps this
    correct for any SwiGLU-style MLP without assuming its exact internal
    wiring.

    Assumes a Llama/Mistral-family model (model.model.layers, each with an
    `.mlp` submodule exposing `.down_proj`).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError(
                "Expected a Llama/Mistral-family model exposing `model.model.layers` "
                "(a ModuleList of decoder layers, each with an `.mlp.down_proj` submodule)."
            )
        self.layers = model.model.layers
        self.outputs: list[torch.Tensor] = []
        self._handles: list[Any] = []

    def _hook(self, module: torch.nn.Module, inputs: Any) -> None:
        tensor = inputs[0] if isinstance(inputs, tuple) else inputs
        self.outputs.append(tensor.detach())

    def __enter__(self) -> "MLPInputCapture":
        self.outputs = []
        self._handles = [
            layer.mlp.down_proj.register_forward_pre_hook(self._hook) for layer in self.layers
        ]
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def neuron_usage(activations: np.ndarray) -> np.ndarray:
    """Mean |activation| per neuron across tokens. activations: (n_tokens,
    d_ff). Returns (d_ff,) — how strongly, on average, each intermediate
    channel fires.
    """
    return np.abs(activations).mean(axis=0)


def cluster_tokens(Psi: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """K-means over diffusion coordinates, standing in for which expert
    DiffusionRouter would assign each token to. Returns (n_tokens,) int
    labels in [0, n_clusters).
    """
    n_clusters = min(n_clusters, Psi.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    return kmeans.fit_predict(Psi)


def neuron_specialization_analysis(
    activations: np.ndarray, labels: np.ndarray, width_k: int
) -> dict[str, Any]:
    """For each cluster, checks how much of its own FFN activation "mass" a
    width_k-neuron slice captures — once using that cluster's OWN top
    neurons (what a genuinely specialized narrow expert would use), and once
    using the single globally-top-activating width_k neurons (what a
    cluster-agnostic narrow slice would give every cluster for free, with no
    specialization at all).

    activations: (n_tokens, d_ff) captured MLP intermediate activations.
    labels: (n_tokens,) cluster assignment from cluster_tokens.
    width_k: the candidate per-expert FFN width being evaluated (e.g.
    d_ff // n_experts, matching expert_ffn.py's actual formula).

    `specialization_gain` per cluster is the key number: own-cluster
    coverage minus what the same-size globally-shared slice already gets for
    free. Near zero means the cluster doesn't need anything a shared narrow
    slice wouldn't already provide — evidence AGAINST the width-reduction
    hypothesis for that cluster, independent of whatever r* says about
    routing. Large and positive means the cluster genuinely leans on neurons
    the global slice would have missed — evidence a specialized narrow
    expert has real capacity to exploit there.

    `pairwise_jaccard` is the Jaccard overlap between clusters' own top-k
    neuron sets: near 1 means clusters aren't actually using different
    capacity (defeats the point of separate narrow experts); near 0 means
    they're relying on largely disjoint neurons (consistent with narrow,
    specialized experts being viable).
    """
    activations = np.asarray(activations)
    n_tokens, d_ff = activations.shape
    width_k = min(width_k, d_ff)
    cluster_ids = sorted(np.unique(labels).tolist())

    population_usage = neuron_usage(activations)
    global_top_k = set(np.argsort(population_usage)[::-1][:width_k].tolist())

    per_cluster: dict[int, dict[str, Any]] = {}
    top_k_sets: dict[int, set[int]] = {}
    for c in cluster_ids:
        cluster_acts = activations[labels == c]
        usage = neuron_usage(cluster_acts)
        total_mass = usage.sum()
        top_k = set(np.argsort(usage)[::-1][:width_k].tolist())
        top_k_sets[c] = top_k

        own_coverage = usage[list(top_k)].sum() / total_mass if total_mass > 0 else 0.0
        global_coverage = usage[list(global_top_k)].sum() / total_mass if total_mass > 0 else 0.0

        per_cluster[c] = {
            "n_tokens": int((labels == c).sum()),
            "own_coverage": float(own_coverage),
            "global_coverage": float(global_coverage),
            "specialization_gain": float(own_coverage - global_coverage),
        }

    pairwise_jaccard: dict[str, float] = {}
    for i, c1 in enumerate(cluster_ids):
        for c2 in cluster_ids[i + 1 :]:
            union = top_k_sets[c1] | top_k_sets[c2]
            inter = top_k_sets[c1] & top_k_sets[c2]
            jaccard = len(inter) / len(union) if union else 0.0
            pairwise_jaccard[f"{c1}-{c2}"] = float(jaccard)

    gains = [v["specialization_gain"] for v in per_cluster.values()]
    jaccards = list(pairwise_jaccard.values())

    return {
        "width_k": width_k,
        "n_clusters": len(cluster_ids),
        "per_cluster": per_cluster,
        "pairwise_jaccard": pairwise_jaccard,
        "mean_specialization_gain": float(np.mean(gains)) if gains else 0.0,
        "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
    }


def oracle_cluster_tokens_by_activation(
    mlp_activations: np.ndarray,
    n_clusters: int,
    n_components: int = 32,
    random_state: int = 42,
) -> np.ndarray:
    """K-means directly on the FFN activations themselves (PCA-reduced to
    n_components first — the same dimensionality diffusion clustering uses,
    for a fair comparison rather than giving the oracle an unfair
    information advantage) — the best-case partition achievable for the
    neuron_specialization_analysis metric, since it groups tokens by
    exactly the signal that metric measures. This is NOT circular in a bad
    way: it's the intended use, establishing a ceiling on how much
    specialization is achievable by ANY K-way partition of these tokens,
    independent of whether diffusion geometry (or anything else) can find
    it. Returns (n_tokens,) int labels in [0, n_clusters).
    """
    mlp_activations = np.asarray(mlp_activations, dtype=np.float64)
    n_components = min(n_components, mlp_activations.shape[0], mlp_activations.shape[1])
    reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(
        mlp_activations
    )
    return cluster_tokens(reduced, n_clusters=n_clusters, random_state=random_state)


def cluster_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float]:
    """Adjusted Rand Index and Normalized Mutual Information between two
    cluster labelings of the SAME tokens (e.g. a diffusion-based clustering
    vs. the oracle activation-based one) — measures how much one partition's
    structure the other recovers, independent of label numbering (both
    metrics are permutation-invariant, so which integer label means what
    doesn't matter).

    ARI ~= 1 / NMI ~= 1: near-identical partitions (diffusion clustering is
    already finding most of the achievable structure). ARI ~= 0: chance-level
    agreement (diffusion clustering isn't finding the oracle's structure at
    all, even though — if oracle specialization is strong — that structure
    exists to be found). ARI can go negative for worse-than-chance agreement;
    NMI is bounded in [0, 1].
    """
    return {
        "adjusted_rand_index": float(adjusted_rand_score(labels_a, labels_b)),
        "normalized_mutual_info": float(normalized_mutual_info_score(labels_a, labels_b)),
    }
