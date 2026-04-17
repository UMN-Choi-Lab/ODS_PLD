"""Offline smoke tests that do NOT require SUMO or BO4Mob.

Run with: pytest -q
"""

from __future__ import annotations

import numpy as np

from odspld import kkt_residual, nnls_solve, projected_langevin


def _toy() -> tuple[np.ndarray, np.ndarray]:
    # 1-origin / 2-destination shared link.
    A = np.array([[1.0, 1.0]])
    y = np.array([10.0])
    return A, y


def test_nnls_on_toy_is_feasible_and_fits() -> None:
    A, y = _toy()
    x = nnls_solve(A, y)
    assert (x >= 0).all()
    assert np.isclose(A @ x, y, atol=1e-6).all()


def test_kkt_residual_vanishes_at_nnls_solution() -> None:
    A, y = _toy()
    x = nnls_solve(A, y)
    kkt = kkt_residual(A, y, x)
    assert kkt["primal_min"] >= -1e-8
    assert kkt["dual_min"] >= -1e-8
    assert kkt["complementary_slackness"] < 1e-6


def test_pld_samples_are_non_negative_and_near_constraint() -> None:
    A, y = _toy()
    samples = projected_langevin(A, y, tau=2.0, N=40, n_steps=20_000, seed=0)
    assert (samples >= 0).all()
    assert np.isclose(samples.sum(axis=1).mean(), 10.0, atol=1.0)


def test_pld_explores_nullspace() -> None:
    A, y = _toy()
    samples = projected_langevin(A, y, tau=2.0, N=200, n_steps=60_000, seed=0)
    spread = samples[:, 0].std()
    # Unconstrained SVD-mode sampling would give samples with x_1 = x_2 (zero
    # null-space variance). PLD should have meaningful spread along the
    # null-space direction (1, -1)^T.
    assert spread > 0.5
