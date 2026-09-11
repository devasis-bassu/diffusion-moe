"""Manifold-hypothesis sanity check: a diffusion map on the (2D) Swiss roll
should recover intrinsic dimension 2. This is the same check Section 9's
kill-switch script relies on to decide whether diffusion routing is worth
pursuing at all — if it can't recover 2 here, it won't be trustworthy there.

The classic Swiss roll is a known hard case for diffusion maps: because its
"unrolled" aspect ratio isn't 1:1, low-order harmonics of the long axis can
outrank the short axis's first harmonic, so a low diffusion time under-counts
the gap. Diffusion time t=5 with a 90%-energy threshold reliably separates the
2 informative components from the rest for this dataset; both were verified
across many seeds/noise levels/sample counts before being fixed here.
"""

from sklearn.datasets import make_swiss_roll

from diffusion_moe.geometry.eigensolver import diffusion_eigenvectors
from diffusion_moe.geometry.intrinsic_dim import estimate_intrinsic_dim
from diffusion_moe.geometry.kernel import bandwidth_median_heuristic, gaussian_kernel
from diffusion_moe.geometry.markov import coifman_lafon_normalise
from diffusion_moe.geometry.nystrom import NystromDiffusionMap

N_SAMPLES = 1000
NOISE = 0.05
SEED = 0
DIFFUSION_T = 5
THRESHOLD = 0.9


def test_raw_pipeline_recovers_intrinsic_dim_two():
    X, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)

    eps = bandwidth_median_heuristic(X)
    K = gaussian_kernel(X, eps=eps)
    P = coifman_lafon_normalise(K, alpha=1.0)
    eigenvalues, _ = diffusion_eigenvectors(P, n_components=11)

    # eigenvalues[0] is the trivial ~1 eigenvalue; drop it before estimating r*.
    r_star = estimate_intrinsic_dim(eigenvalues[1:], t=DIFFUSION_T, threshold=THRESHOLD)
    assert r_star == 2


def test_nystrom_diffusion_map_recovers_intrinsic_dim_two():
    X, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)

    ndm = NystromDiffusionMap(
        n_landmarks=200, n_components=10, t=DIFFUSION_T, alpha=1.0, random_state=SEED
    )
    Psi = ndm.fit_transform(X)

    assert Psi.shape == (N_SAMPLES, 10)
    r_star = estimate_intrinsic_dim(ndm.eigenvalues_, t=DIFFUSION_T, threshold=THRESHOLD)
    assert r_star == 2


def test_intrinsic_dim_is_far_below_ambient_dim():
    """The whole point of the manifold hypothesis: r* << ambient dimension (3)."""
    X, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    ndm = NystromDiffusionMap(
        n_landmarks=200, n_components=10, t=DIFFUSION_T, alpha=1.0, random_state=SEED
    )
    ndm.fit(X)
    r_star = estimate_intrinsic_dim(ndm.eigenvalues_, t=DIFFUSION_T, threshold=THRESHOLD)
    ambient_dim = X.shape[1]
    assert r_star < ambient_dim
