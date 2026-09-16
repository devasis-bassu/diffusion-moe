"""Nystrom-extended diffusion maps for out-of-sample extension."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from diffusion_moe.geometry.eigensolver import diffusion_eigenvectors
from diffusion_moe.geometry.kernel import (
    bandwidth_median_heuristic,
    gaussian_kernel,
    gaussian_kernel_cross,
)
from diffusion_moe.geometry.markov import coifman_lafon_normalise


class NystromDiffusionMap:
    """Fits diffusion-map eigenfunctions on a small landmark set (selected via
    k-means++), then extends them to new points via the Nystrom formula. This
    avoids ever forming the full O(n^2) kernel matrix over all tokens.

    The trivial top eigenpair of the Markov matrix (eigenvalue ~1, the constant
    eigenvector) carries no geometric information and is discarded internally,
    so `eigenvalues_`/`eigenvectors_` and the returned coordinates directly index
    the informative diffusion directions: eigenvalues_[0] is the largest
    non-trivial eigenvalue, etc.
    """

    def __init__(
        self,
        n_landmarks: int = 128,
        n_components: int = 32,
        t: int = 3,
        alpha: float = 1.0,
        random_state: int = 42,
        eps: float | None = None,
    ) -> None:
        self.n_landmarks = n_landmarks
        self.n_components = n_components
        self.t = t
        self.alpha = alpha
        self.random_state = random_state
        # Overrides the median-heuristic bandwidth with a fixed value when
        # set, instead of deriving eps_ from the landmarks at fit() time.
        # Lets callers hold everything else (landmarks included, since
        # k-means with a fixed random_state on the same Z is deterministic)
        # fixed while sweeping only the kernel scale — see
        # geometry/multiscale.py, which needs exactly that to isolate what
        # changes as eps varies from what changes as the landmarks do.
        self.eps = eps

        self.landmarks_: np.ndarray | None = None
        self.eps_: float | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.eigenvectors_: np.ndarray | None = None
        self.psi_landmarks_: np.ndarray | None = None
        self._d_alpha_landmarks: np.ndarray | None = None

    def fit(self, Z: np.ndarray) -> "NystromDiffusionMap":
        Z = np.asarray(Z, dtype=np.float64)
        n = Z.shape[0]
        n_landmarks = min(self.n_landmarks, n)

        kmeans = KMeans(
            n_clusters=n_landmarks,
            init="k-means++",
            n_init=10,
            random_state=self.random_state,
        )
        kmeans.fit(Z)
        self.landmarks_ = kmeans.cluster_centers_

        self.eps_ = self.eps if self.eps is not None else bandwidth_median_heuristic(
            self.landmarks_
        )
        K = gaussian_kernel(self.landmarks_, self.eps_)
        P = coifman_lafon_normalise(K, alpha=self.alpha)

        # Solve for one extra eigenpair so we can drop the trivial top one
        # (eigenvalue ~1) and still return n_components informative directions.
        n_solve = min(self.n_components + 1, n_landmarks - 1)
        eigenvalues, eigenvectors = diffusion_eigenvectors(P, n_components=n_solve)

        self.eigenvalues_ = eigenvalues[1:]
        self.eigenvectors_ = eigenvectors[:, 1:]
        # Landmarks are the points P was built from, so their own diffusion
        # coordinates are the direct (non-Nystrom) formula: eigenvector scaled
        # by eigenvalue^t — no kernel extension needed, unlike transform().
        self.psi_landmarks_ = self.eigenvectors_ * (self.eigenvalues_[None, :] ** self.t)

        d_landmarks = K.sum(axis=1)
        self._d_alpha_landmarks = np.power(d_landmarks, self.alpha)

        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Extends diffusion coordinates to new points Z via the Nystrom formula.

        Returns Psi_t of shape (n, n_components), where
        Psi_t[:, i] = lambda_i^t * phi_i(Z) and phi_i is the Nystrom-extended
        i-th eigenfunction of the landmark Markov matrix.
        """
        if self.landmarks_ is None:
            raise RuntimeError("Call fit(Z) or fit_transform(Z) before transform().")

        Z = np.asarray(Z, dtype=np.float64)
        K_new = gaussian_kernel_cross(Z, self.landmarks_, self.eps_)

        d_new = K_new.sum(axis=1)
        d_alpha_new = np.power(d_new, self.alpha)
        # A point whose kernel mass to every landmark underflows to exactly 0
        # (drifted entirely outside the landmarks' bandwidth -- observed in
        # practice once training pushes embeddings far enough between refits)
        # would otherwise give 0/0 = NaN here, silently, since this is a raw
        # numpy division with no validation. Guarded the same way row_sums is
        # guarded below: such a point gets kernel weight 0 to every landmark
        # instead of NaN: P_new normalizes to zero rows too, so
        # Psi_t for it is defined as the origin, i.e. maximal uncertainty
        # about its diffusion coordinates. That's a reasonable fallback --
        # this NaN is not.
        d_alpha_new_safe = np.where(d_alpha_new == 0, 1.0, d_alpha_new)
        K_alpha = K_new / (d_alpha_new_safe[:, None] * self._d_alpha_landmarks[None, :])

        row_sums = K_alpha.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        P_new = K_alpha / row_sums  # (n, n_landmarks)

        eigenvalues_safe = np.where(
            np.abs(self.eigenvalues_) < 1e-12, 1e-12, self.eigenvalues_
        )
        phi_new = (P_new @ self.eigenvectors_) / eigenvalues_safe[None, :]

        Psi_t = phi_new * (self.eigenvalues_[None, :] ** self.t)
        return Psi_t

    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        self.fit(Z)
        return self.transform(Z)
