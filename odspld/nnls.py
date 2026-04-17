"""Non-Negative Least Squares solver and KKT-condition checker."""

from __future__ import annotations

import numpy as np
import scipy.optimize as opt


def nnls_solve(A: np.ndarray, y: np.ndarray, max_iter: int = 5000) -> np.ndarray:
    """Solve min_{x >= 0} (1/2) ||A x - y||^2.

    Uses scipy's bounded variable least squares (BVLS, Lawson--Hanson).
    """
    result = opt.lsq_linear(A, y, bounds=(0.0, np.inf), method="bvls", max_iter=max_iter)
    return result.x


def kkt_residual(A: np.ndarray, y: np.ndarray, x: np.ndarray) -> dict:
    """Check the NNLS KKT conditions: x >= 0, r >= 0, and x_i * r_i = 0."""
    r = A.T @ (A @ x - y)
    return {
        "primal_min": float(x.min()),
        "dual_min": float(r.min()),
        "complementary_slackness": float(np.max(np.abs(x * r))),
    }
