"""Multiscale (dyadic) diffusion map analysis.

A single kernel bandwidth eps commits to one notion of "nearby" — exactly
the failure mode that made the Phase-1 kill switch (scripts/extract_geometry.py)
report a spuriously tiny r* on real Mistral-7B activations: the
median-heuristic eps locked onto the dense bulk of tokens while leaving a
handful of outlier-norm tokens (e.g. attention-sink / massive-activation
positions, well documented for Llama/Mistral-family models) disconnected
from the rest of the kernel graph, which trivially collapses r* without
reflecting any real low-dimensional structure.

Swiss-roll-style manifolds only "unfold" in a bounded window of scales: too
small an eps and the kernel graph over-fragments into near-isolated
micro-clusters; too large and it collapses into one connected blob with no
structure left to resolve. Rather than commit to one eps (or trust that the
median heuristic found the right one), this sweeps a dyadic ladder of
bandwidths and lets each scale's r*/spectral-gap/eigenvalue profile reveal
whether a stable "unfolded" regime exists at all, and where.

Operates on cosine (angular) distance by default rather than raw Euclidean
distance: for unit vectors, ||x-y||^2 = 2 - 2*cos(x,y), so normalizing rows
to unit norm before kernelizing is a monotonic reparametrization from cosine
to Euclidean distance, AND it discards each token's raw activation norm from
the metric entirely — the exact axis that makes attention-sink tokens
catastrophic outliers under a raw-Euclidean kernel.
"""

from __future__ import annotations

import numpy as np

from diffusion_moe.geometry.intrinsic_dim import estimate_intrinsic_dim, spectral_gap
from diffusion_moe.geometry.kernel import bandwidth_median_heuristic
from diffusion_moe.geometry.nystrom import NystromDiffusionMap


def l2_normalize(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalizes Z to unit norm.

    Squared Euclidean distance between unit vectors is a monotonic
    reparametrization of cosine distance (||x-y||^2 = 2 - 2*cos(x,y)), so a
    Gaussian kernel on the result is a cosine/angular kernel. eps guards
    against dividing by a near-zero norm.
    """
    Z = np.asarray(Z, dtype=np.float64)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.clip(norms, eps, None)


def dyadic_eps_ladder(eps_center: float, n_scales: int = 7, base: float = 2.0) -> list[float]:
    """Symmetric dyadic ladder of bandwidths around eps_center: eps_center *
    base^k for k in [-n_scales//2, ..., n_scales//2].

    n_scales should be odd so the ladder is centered on eps_center (typically
    the median-heuristic estimate) with an equal number of finer/coarser
    scales on either side.
    """
    if n_scales < 1:
        raise ValueError("n_scales must be >= 1")
    if eps_center <= 0:
        raise ValueError(f"eps_center must be positive, got {eps_center}")
    half = n_scales // 2
    return [eps_center * (base**k) for k in range(-half, half + 1)]


def multiscale_diffusion_analysis(
    Z: np.ndarray,
    n_landmarks: int = 128,
    n_components: int = 32,
    t: int = 3,
    alpha: float = 1.0,
    n_scales: int = 7,
    base: float = 2.0,
    cosine: bool = True,
    random_state: int = 42,
) -> dict:
    """Fits a NystromDiffusionMap at each scale in a dyadic ladder around the
    median-heuristic bandwidth of Z, holding everything except eps fixed
    across scales — landmarks are chosen once via k-means with a fixed
    random_state, which is deterministic given the same Z, so re-fitting at
    each eps effectively reselects the same landmarks and isolates what
    changes with scale alone.

    Returns a dict with the eps ladder (in both raw and eps/eps_center-ratio
    form) and, per scale: r*, spectral gap, top eigenvalues, and whether the
    leading non-trivial eigenvalue is ~1 (the disconnection signature) — the
    minimum needed to read off whether there's a stable "unfolded" window,
    or whether the manifold looks fragmented at every scale tested.
    """
    Z = l2_normalize(Z) if cosine else np.asarray(Z, dtype=np.float64)
    eps_center = bandwidth_median_heuristic(Z)
    eps_values = dyadic_eps_ladder(eps_center, n_scales=n_scales, base=base)

    scales = []
    for eps in eps_values:
        ndm = NystromDiffusionMap(
            n_landmarks=n_landmarks,
            n_components=n_components,
            t=t,
            alpha=alpha,
            random_state=random_state,
            eps=eps,
        )
        ndm.fit(Z)
        r_star = estimate_intrinsic_dim(ndm.eigenvalues_, t=t)
        delta = spectral_gap(ndm.eigenvalues_)
        scales.append(
            {
                "eps": eps,
                "eps_ratio_to_median": eps / eps_center,
                "r_star": r_star,
                "spectral_gap": delta,
                "top_eigenvalues": ndm.eigenvalues_[:5].tolist(),
                "likely_disconnected": bool(ndm.eigenvalues_[0] > 0.999),
            }
        )

    return {"eps_center": eps_center, "cosine": cosine, "scales": scales}


def find_stable_window(scales: list[dict], min_run: int = 2) -> list[int]:
    """Returns the indices (into `scales`, as returned by
    multiscale_diffusion_analysis) of the longest consecutive run where r* is
    constant AND not flagged likely_disconnected — i.e. the longest run of
    adjacent scales that agree on the same, non-degenerate intrinsic
    dimension. Coifman & Maggioni-style multiscale analysis reads intrinsic
    dimension off a plateau like this rather than trusting any single scale;
    an empty list means no such plateau of length >= min_run was found (the
    manifold looks fragmented or unstable at every scale tested).
    """
    best: list[int] = []
    run: list[int] = []
    for i, s in enumerate(scales):
        if (
            run
            and not s["likely_disconnected"]
            and s["r_star"] == scales[run[-1]]["r_star"]
        ):
            run.append(i)
        elif not s["likely_disconnected"]:
            run = [i]
        else:
            run = []
        if len(run) > len(best):
            best = list(run)
    return best if len(best) >= min_run else []
