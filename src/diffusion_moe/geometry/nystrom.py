"""Nystrom-extended diffusion maps for out-of-sample extension."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from diffusion_moe.geometry.eigensolver import diffusion_eigenvectors
from diffusion_moe.geometry.kernel import bandwidth_median_heuristic, gaussian_kernel
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

        # Diffusion-map eigenvectors are only defined up to sign (Nystrom's
        # own docstring already notes eigenvalues are non-negative; signs
        # are the other half of that ambiguity). A refit doesn't perturb the
        # old basis, it replaces it outright -- fresh landmarks (a new
        # k-means draw) and a fresh eigensolve, with no guaranteed
        # relationship to the old basis's orientation. ExpertCentroids
        # (models/moe_layer.py) is an ordinary nn.Parameter trained by
        # gradient descent against whatever basis was in effect -- it does
        # NOT get remapped when this happens, so an arbitrary sign flip here
        # silently scrambles routing relative to it. Confirmed empirically,
        # not just in theory (phase2_training_report.md): load_loss crashes
        # to ~0 immediately after every refit and takes ~10-15 steps to
        # recover, with a correlated task_loss spike -- the router
        # re-learning alignment it had already learned once, every single
        # refit. Snapshot the OLD basis's embedding of this same batch
        # before overwriting anything, so the new basis can be aligned to it
        # below -- None on the very first fit (nothing to align against yet).
        psi_old = self.transform(Z, _refresh_eps=False) if self.landmarks_ is not None else None

        kmeans = KMeans(
            n_clusters=n_landmarks,
            init="k-means++",
            n_init=10,
            random_state=self.random_state,
        )
        kmeans.fit(Z)
        self.landmarks_ = kmeans.cluster_centers_

        eps = self.eps if self.eps is not None else bandwidth_median_heuristic(self.landmarks_)
        self._fit_landmark_spectrum(eps)

        if psi_old is not None:
            psi_new = self.transform(Z, _refresh_eps=False)
            self._align_sign_to(psi_old, psi_new)

        return self

    def _align_sign_to(self, psi_old: np.ndarray, psi_new: np.ndarray) -> None:
        """Flips each component's sign in the just-fit eigenbasis to maximize
        agreement with psi_old (the previous basis's embedding of the same
        batch), component by component. Landmarks themselves changed (a
        fresh k-means draw), so there's no per-point correspondence to align
        directly -- comparing both bases' embeddings of the same batch Z is
        the common reference that makes "agreement" measurable at all.

        Sign-only: this does not correct rotation within a near-degenerate
        eigenspace (two eigenvalues close enough that their eigenvectors can
        mix, not just flip) -- a real but smaller-scope gap than the sign
        ambiguity this does fix, left as a follow-up if this alone doesn't
        meaningfully reduce the refit disruption.
        """
        for k in range(psi_old.shape[1]):
            if np.dot(psi_old[:, k], psi_new[:, k]) < 0:
                self.eigenvectors_[:, k] *= -1
                self.psi_landmarks_[:, k] *= -1

    def _fit_landmark_spectrum(self, eps: float) -> None:
        """(Re)derives everything that depends on the kernel bandwidth, given
        the current (fixed) `self.landmarks_`: the landmark-landmark kernel
        and Markov matrix, its eigendecomposition, and each landmark's own
        direct (non-Nystrom) diffusion coordinates.

        Called by fit() after a fresh k-means, and by transform()'s eps_
        live-refresh (see there) to keep eigenvectors_/psi_landmarks_
        consistent with whatever eps_ is currently in effect -- required for
        Nystrom's defining correctness property (transform() applied to the
        landmarks themselves must reproduce psi_landmarks_ exactly); reusing
        stale eigenvectors_ computed from the *old* eps_ silently breaks that.
        This is an eigensolve over an n_landmarks x n_landmarks matrix --
        O(n_landmarks^3), negligible next to the O(n_landmarks) k-means
        clustering over the full token batch that fit() alone does and this
        does not repeat.
        """
        n_landmarks = self.landmarks_.shape[0]
        self.eps_ = eps
        K = gaussian_kernel(self.landmarks_, self.eps_)
        P = coifman_lafon_normalise(K, alpha=self.alpha)

        # Solve for one extra eigenpair so we can drop the trivial top one
        # (eigenvalue ~1) and still return n_components informative directions.
        n_solve = min(self.n_components + 1, n_landmarks - 1)
        eigenvalues, eigenvectors = diffusion_eigenvectors(
            P, n_components=n_solve, random_state=self.random_state
        )

        self.eigenvalues_ = eigenvalues[1:]
        self.eigenvectors_ = eigenvectors[:, 1:]
        # Landmarks are the points P was built from, so their own diffusion
        # coordinates are the direct (non-Nystrom) formula: eigenvector scaled
        # by eigenvalue^t — no kernel extension needed, unlike transform().
        self.psi_landmarks_ = self.eigenvectors_ * (self.eigenvalues_[None, :] ** self.t)

        d_landmarks = K.sum(axis=1)
        self._d_alpha_landmarks = np.power(d_landmarks, self.alpha)

    def transform(self, Z: np.ndarray, _refresh_eps: bool = True) -> np.ndarray:
        """Extends diffusion coordinates to new points Z via the Nystrom formula.

        Returns Psi_t of shape (n, n_components), where
        Psi_t[:, i] = lambda_i^t * phi_i(Z) and phi_i is the Nystrom-extended
        i-th eigenfunction of the landmark Markov matrix.

        `_refresh_eps` is internal (see fit_transform): a standalone call --
        the real use case, e.g. models/moe_layer.py calling this on every
        step *between* landmark refits -- lets eps_ track this batch's actual
        distance to the (still-frozen) landmarks. fit_transform's own
        internal call disables it, since it runs immediately after fit() on
        the very same Z: there's no staleness yet to correct, and
        refreshing anyway would silently override fit()'s landmark-based
        heuristic with a different (batch-based) one for no benefit, plus
        redundantly redo the eigensolve _fit_landmark_spectrum below.
        """
        if self.landmarks_ is None:
            raise RuntimeError("Call fit(Z) or fit_transform(Z) before transform().")

        Z = np.asarray(Z, dtype=np.float64)
        sq_dists = cdist(Z, self.landmarks_, metric="sqeuclidean")

        if _refresh_eps and self.eps is None:
            # self.eps_ was set from the landmarks' own spread at the last
            # fit() -- everything else about the landmarks/eigenvectors stays
            # fixed until the next refit (expensive: k-means + eigensolve),
            # but that leaves eps_ describing a distribution that's
            # increasingly stale the longer training runs between refits.
            # Re-deriving it here from *this batch's* actual distance to the
            # (still-frozen) landmarks is cheap -- reuses sq_dists, already
            # computed above for K_new -- and keeps the kernel bandwidth
            # tracking real distributional drift instead of a step-0
            # snapshot. This is what actually prevents the failure mode
            # d_alpha_new_safe below merely papers over: a point drifting far
            # enough, relative to a *stale* eps_, that every kernel value
            # underflows to exactly 0.
            #
            # Must go through _fit_landmark_spectrum (not just eps_ and
            # _d_alpha_landmarks in isolation): eigenvectors_/psi_landmarks_
            # were computed from the *old* eps_ at fit() time, and Nystrom's
            # correctness guarantee only holds when the landmark spectrum and
            # the new-point extension share one consistent kernel. That's an
            # eigensolve over an n_landmarks x n_landmarks matrix, not
            # another k-means -- still cheap relative to the
            # O(n_tokens * n_landmarks * d) cost already paid above for
            # sq_dists.
            eps_candidate = float(np.median(sq_dists))
            eps = eps_candidate if eps_candidate > 0 else float(np.finfo(np.float64).eps)
            self._fit_landmark_spectrum(eps)

        K_new = np.exp(-sq_dists / self.eps_)

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
        return self.transform(Z, _refresh_eps=False)
