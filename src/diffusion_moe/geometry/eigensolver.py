"""Eigendecomposition of the diffusion Markov matrix."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig as dense_eig
from scipy.sparse.linalg import ArpackNoConvergence, eigs


def diffusion_eigenvectors(
    P: np.ndarray, n_components: int = 32, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Top eigenvalues/eigenvectors of the Markov matrix P, sorted descending.

    Uses ARPACK (scipy.sparse.linalg.eigs) for efficiency, since we only need the
    top-k eigenpairs of a matrix that may be too large to fully diagonalise. Falls
    back to a dense solve when the matrix is too small for ARPACK's k < n - 1
    constraint, OR when ARPACK's iterative Lanczos method fails to converge —
    hit in practice during a real 32-layer multiscale bandwidth sweep (a very
    small or very large kernel bandwidth can leave P with a near-degenerate or
    tightly clustered eigenspectrum, which is exactly what makes ARPACK's
    iterative method struggle to converge in the first place). The matrix
    sizes here are landmark counts (n_landmarks, at most a few hundred), so a
    dense fallback is cheap regardless of why it's taken.

    P's eigenvalues are real and non-negative: the Coifman-Lafon-normalised chain
    is reversible, so P is similar to a symmetric PSD matrix even though P itself
    is only row-stochastic (not symmetric).

    Returns (eigenvalues, eigenvectors) with eigenvectors[:, i] corresponding to
    eigenvalues[i].
    """
    n = P.shape[0]
    k = min(n_components, n - 2)

    # ARPACK's Lanczos iteration starts from a random vector when v0 isn't
    # given, so two calls on the *same* P can converge to a different (but
    # equally valid, e.g. sign-flipped) eigenvector basis. Harmless when this
    # only ran once per landmark set, but nystrom.py's transform() now reruns
    # this every time its live eps_ refresh fires -- an unseeded v0 would
    # make Psi_t's basis jitter step to step even without real drift in the
    # data, which is exactly the routing instability this was meant to fix,
    # not introduce. Fixing v0 removes that: identical P -> identical result.
    v0 = np.random.RandomState(random_state).rand(n) if random_state is not None else None

    if k >= 1 and k < n - 1:
        try:
            eigenvalues, eigenvectors = eigs(P, k=k, which="LM", v0=v0)
        except ArpackNoConvergence:
            eigenvalues, eigenvectors = dense_eig(P)
    else:
        eigenvalues, eigenvectors = dense_eig(P)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][:n_components]
    eigenvectors = eigenvectors[:, order][:, :n_components]

    return eigenvalues, eigenvectors
