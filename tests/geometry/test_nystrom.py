"""Tests for NystromDiffusionMap."""

import numpy as np

from diffusion_moe.geometry.kernel import bandwidth_median_heuristic
from diffusion_moe.geometry.nystrom import NystromDiffusionMap


def _make_data(n=200, seed=0):
    return np.random.RandomState(seed).randn(n, 5)


def test_fit_transform_shape():
    Z = _make_data(n=200)
    ndm = NystromDiffusionMap(n_landmarks=32, n_components=10, t=3, alpha=1.0)
    Psi = ndm.fit_transform(Z)
    assert Psi.shape == (200, 10)
    assert np.all(np.isfinite(Psi))


def test_eigenvalues_and_eigenvectors_exclude_trivial_component():
    Z = _make_data(n=200)
    ndm = NystromDiffusionMap(n_landmarks=32, n_components=10, t=3, alpha=1.0)
    ndm.fit(Z)
    assert ndm.eigenvalues_.shape == (10,)
    assert ndm.eigenvectors_.shape == (32, 10)
    # the trivial eigenvalue (~1) has been dropped
    assert ndm.eigenvalues_[0] < 1.0 - 1e-6


def test_landmarks_clipped_to_n_when_fewer_points_than_landmarks():
    Z = _make_data(n=10)
    ndm = NystromDiffusionMap(n_landmarks=128, n_components=3, t=1, alpha=1.0)
    ndm.fit(Z)
    assert ndm.landmarks_.shape[0] == 10


def test_transform_before_fit_raises():
    ndm = NystromDiffusionMap(n_landmarks=32, n_components=10)
    import pytest

    with pytest.raises(RuntimeError):
        ndm.transform(_make_data(n=5))


def test_fit_transform_uses_landmark_based_eps_without_live_refresh():
    """fit_transform's own internal transform() call must NOT trigger the
    eps_ live-refresh (Option 1, see transform()'s _refresh_eps param):
    immediately after fit(), eps_ already reflects this exact landmark set
    with no staleness to correct, and refreshing anyway would silently swap
    out fit()'s landmark-based median heuristic for a different (batch-based)
    number, plus redundantly redo the eigensolve, on every refit step."""
    Z = _make_data(n=150)
    ndm = NystromDiffusionMap(n_landmarks=40, n_components=5, t=2, alpha=1.0)
    ndm.fit_transform(Z)
    assert ndm.eps_ == bandwidth_median_heuristic(ndm.landmarks_)


def test_transform_is_idempotent_on_repeated_calls_with_same_data():
    """A standalone transform() call DOES live-refresh eps_ (unlike
    fit_transform's internal one, see above) -- but calling it twice in a
    row on the *same* data must give the same answer both times: identical
    Z means identical sq_dists means identical refreshed eps_, and
    diffusion_eigenvectors' ARPACK solve is now seeded (random_state),
    so identical inputs give a bit-identical eigenbasis rather than an
    arbitrary sign-flipped one on each call."""
    Z = _make_data(n=150)
    ndm = NystromDiffusionMap(n_landmarks=40, n_components=5, t=2, alpha=1.0, random_state=0)
    ndm.fit(Z)
    Psi_a = ndm.transform(Z)
    Psi_b = ndm.transform(Z)
    assert np.allclose(Psi_a, Psi_b)


def test_eps_override_skips_median_heuristic():
    Z = _make_data(n=200)
    auto = NystromDiffusionMap(n_landmarks=32, n_components=10, random_state=0)
    auto.fit(Z)

    fixed = NystromDiffusionMap(n_landmarks=32, n_components=10, random_state=0, eps=123.0)
    fixed.fit(Z)

    assert fixed.eps_ == 123.0
    assert fixed.eps_ != auto.eps_


def test_eps_override_same_seed_reuses_same_landmarks():
    """A fixed random_state makes k-means deterministic given the same Z, so
    varying only `eps` (as geometry/multiscale.py's dyadic sweep does)
    shouldn't also perturb which points were chosen as landmarks."""
    Z = _make_data(n=200)
    ndm_a = NystromDiffusionMap(n_landmarks=32, n_components=10, random_state=0, eps=1.0)
    ndm_b = NystromDiffusionMap(n_landmarks=32, n_components=10, random_state=0, eps=50.0)
    ndm_a.fit(Z)
    ndm_b.fit(Z)
    np.testing.assert_array_equal(ndm_a.landmarks_, ndm_b.landmarks_)


def test_align_sign_to_flips_eigenvectors_and_psi_landmarks_on_disagreement():
    """Direct, deterministic test of the core alignment logic, decoupled
    from the noisy randomness of a real k-means/eigensolve pipeline: given
    a reference embedding and a "new" one that disagrees in sign on some
    components and agrees on others, only the disagreeing components'
    eigenvectors_/psi_landmarks_ should flip."""
    Z = _make_data(n=100)
    ndm = NystromDiffusionMap(n_landmarks=16, n_components=4, random_state=0)
    ndm.fit(Z)

    eigenvectors_before = ndm.eigenvectors_.copy()
    psi_landmarks_before = ndm.psi_landmarks_.copy()

    n_points = 20
    psi_old = np.random.RandomState(1).randn(n_points, 4)
    psi_new = psi_old.copy()
    psi_new[:, 1] *= -1  # disagreement on component 1 only
    psi_new[:, 3] *= -1  # and component 3

    ndm._align_sign_to(psi_old, psi_new)

    for k in (0, 2):  # agreeing components: untouched
        assert np.allclose(ndm.eigenvectors_[:, k], eigenvectors_before[:, k])
        assert np.allclose(ndm.psi_landmarks_[:, k], psi_landmarks_before[:, k])
    for k in (1, 3):  # disagreeing components: flipped
        assert np.allclose(ndm.eigenvectors_[:, k], -eigenvectors_before[:, k])
        assert np.allclose(ndm.psi_landmarks_[:, k], -psi_landmarks_before[:, k])


def test_first_fit_does_not_attempt_alignment():
    """landmarks_ starts None -- there's no previous basis to align against,
    and _align_sign_to must not be called at all on the very first fit."""
    Z = _make_data(n=100)
    ndm = NystromDiffusionMap(n_landmarks=16, n_components=4, random_state=0)

    calls = []
    ndm._align_sign_to = lambda *a, **kw: calls.append(1)
    ndm.fit(Z)

    assert calls == []


def test_refit_aligns_new_basis_to_agree_with_the_old_ones_embedding():
    """The actual invariant the fix exists for: after a refit on evolved
    (but related) data, the new basis's embedding of a fixed reference batch
    should correlate *positively* with what the old basis said about the
    same batch, component by component -- not land on an arbitrary sign
    per refit, which is what silently scrambles routing relative to
    ExpertCentroids (a plain nn.Parameter that doesn't get remapped)."""
    Z1 = _make_data(n=150, seed=0)
    reference = _make_data(n=30, seed=99)

    ndm = NystromDiffusionMap(n_landmarks=24, n_components=5, random_state=0)
    ndm.fit(Z1)
    psi_before = ndm.transform(reference, _refresh_eps=False)

    Z2 = Z1 + np.random.RandomState(2).randn(*Z1.shape) * 0.5  # evolved data
    ndm.fit(Z2)
    psi_after = ndm.transform(reference, _refresh_eps=False)

    for k in range(psi_before.shape[1]):
        assert np.dot(psi_before[:, k], psi_after[:, k]) >= 0


def test_eps_refreshes_across_transform_calls_when_no_override():
    """Option 1 fix for the staleness bug: eps_ (the kernel bandwidth) should
    track the *current* batch's actual distance to the frozen landmarks,
    refreshed on every transform() call, rather than staying pinned to
    whatever the landmarks looked like at the last fit()."""
    Z = _make_data(n=200)
    ndm = NystromDiffusionMap(n_landmarks=32, n_components=10, random_state=0)
    ndm.fit(Z)
    eps_after_fit = ndm.eps_

    ndm.transform(_make_data(n=50, seed=1))
    eps_near = ndm.eps_

    # A batch shifted far from the landmarks has much larger distances to
    # them, so the median-based eps_ should shift accordingly.
    far_batch = _make_data(n=50, seed=2) + 500.0
    ndm.transform(far_batch)
    eps_far = ndm.eps_

    assert eps_near != eps_after_fit  # refreshed at all, away from the fit-time value
    assert eps_far > eps_near * 10  # tracks the shifted batch's real distances


def test_eps_override_persists_through_transform_calls():
    """An explicit eps override (used by geometry/multiscale.py's dyadic
    sweep to isolate what changes as eps varies from what changes as the
    landmarks do) must not be silently clobbered by the new live-refresh
    behavior -- transform() should leave self.eps_ untouched when self.eps
    was explicitly set."""
    Z = _make_data(n=200)
    ndm = NystromDiffusionMap(n_landmarks=32, n_components=10, random_state=0, eps=123.0)
    ndm.fit(Z)
    assert ndm.eps_ == 123.0

    ndm.transform(_make_data(n=50, seed=1) + 500.0)  # would shift eps_ a lot if not gated
    assert ndm.eps_ == 123.0


def test_transform_outlier_point_in_a_mostly_normal_batch_gives_finite_not_nan():
    """Real bug found via an all-layers training run: a point that drifts far
    enough from the fitted landmarks (plausible mid-training, once centroid
    separation pressure has moved embeddings for several hundred steps since
    the last landmark refit) gets exactly-zero kernel mass to every landmark
    -- d_alpha_new underflows to 0.0, and the unguarded division at line ~111
    used to produce 0/0 = NaN here, which then propagated into task_loss/
    load_loss and, a few steps later, poisoned every model parameter via
    clip_grad_norm_'s inability to repair a NaN gradient.

    Regression test for the d_alpha_new_safe fallback specifically -- as a
    second line of defense on top of the eps_ live-refresh (Option 1): a
    single outlier token in an otherwise-normal batch (the realistic case --
    thousands of normal tokens, a handful of drifted ones) doesn't move the
    batch median enough to rescue it the way a batch of *only* far points
    would."""
    Z = _make_data(n=200)
    ndm = NystromDiffusionMap(n_landmarks=32, n_components=10, t=3, alpha=1.0, random_state=0)
    ndm.fit(Z)

    mostly_normal_batch = _make_data(n=200, seed=3)
    mostly_normal_batch[0] = 1e8  # one token drifted far outside the rest
    Psi = ndm.transform(mostly_normal_batch)

    assert np.all(np.isfinite(Psi))


def test_eigenvalues_decrease_diffusion_time_shrinks_coordinates():
    Z = _make_data(n=150)
    ndm_t1 = NystromDiffusionMap(n_landmarks=40, n_components=5, t=1, alpha=1.0, random_state=0)
    ndm_t5 = NystromDiffusionMap(n_landmarks=40, n_components=5, t=5, alpha=1.0, random_state=0)
    Psi_t1 = ndm_t1.fit_transform(Z)
    Psi_t5 = ndm_t5.fit_transform(Z)
    # eigenvalues < 1, so raising diffusion time shrinks coordinate magnitude
    assert np.abs(Psi_t5).mean() < np.abs(Psi_t1).mean()
