"""Non-GP baselines used as sanity checks against BO4Mob's numbers.

The paper table's `Sobol init`, `SPSA`, `VBO`, `SAASBO`, and `TuRBO` columns come
from BO4Mob itself; here we provide light reimplementations of the simpler
baselines (Sobol init best-of-N and SPSA) so the repo can be sanity-checked
without bringing up BoTorch/GPyTorch when you do not need the big GP baselines.
"""

from __future__ import annotations

import numpy as np


def sobol_init_best(
    n: int,
    lb: float,
    ub: float,
    A: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Return the Sobol-quasi-random sample with the smallest ||Ax - y||^2."""
    from torch.quasirandom import SobolEngine

    engine = SobolEngine(n, scramble=True, seed=seed)
    U = engine.draw(n_samples).numpy()
    X = lb + (ub - lb) * U
    residuals = X @ A.T - y
    gA = np.einsum("ij,ij->i", residuals, residuals)
    return X[int(np.argmin(gA))]


def spsa(
    A: np.ndarray,
    y: np.ndarray,
    n: int,
    lb: float,
    ub: float,
    n_iters: int,
    seed: int = 42,
) -> np.ndarray:
    """Simultaneous Perturbation Stochastic Approximation on g^A(x) under box bounds."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(lb, ub, n)
    A_p = max(1, n_iters // 10)
    a = round(0.05 * ((1 + A_p) ** 0.602), 2)
    best_x = x.copy()
    best_gA = float((A @ x - y) @ (A @ x - y))
    for k in range(1, n_iters + 1):
        ak = a / ((k + A_p) ** 0.602)
        ck = 0.1 / (k ** 0.101)
        d = rng.choice([-1, 1], size=n).astype(float)
        xp = np.clip(x + ck * d, lb, ub)
        xm = np.clip(x - ck * d, lb, ub)
        gp = float((A @ xp - y) @ (A @ xp - y))
        gm = float((A @ xm - y) @ (A @ xm - y))
        x = np.clip(x - ak * (gp - gm) / (2.0 * ck * d), lb, ub)
        g = float((A @ x - y) @ (A @ x - y))
        if g < best_gA:
            best_gA = g
            best_x = x.copy()
    return best_x
