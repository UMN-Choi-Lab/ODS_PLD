"""Projected Langevin Dynamics for f(x) propto exp(-g^A(x)/tau) * 1[x >= 0]."""

from __future__ import annotations

import numpy as np

from .nnls import nnls_solve


def projected_langevin(
    A: np.ndarray,
    y: np.ndarray,
    tau: float = 2.0,
    N: int = 50,
    n_steps: int | None = None,
    warmup_frac: float = 0.25,
    seed: int = 42,
    x_init: np.ndarray | None = None,
    include_init: bool = True,
) -> np.ndarray:
    """Sample from the non-negative surrogate posterior via PLD.

    Update rule (step size eta chosen below the stability limit):
        x_{t+1} = max(0, x_t - eta * 2 A^T (A x_t - y) + sqrt(2 eta tau) z_t).

    Returns an (N, n) array of correlated posterior samples. When
    `include_init=True` (default) the first row is the initial point (NNLS
    solution unless `x_init` is given) and the remaining N-1 rows are thinned
    draws from the chain. This guarantees that best-of-N selection can never
    do worse than evaluating the NNLS mode alone, which is the claim the paper
    table relies on ("PLD best-N equals NNLS by construction"). Set
    `include_init=False` for a pure-chain output.
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    if n_steps is None:
        n_steps = max(50_000, N * 500)

    ATA = A.T @ A
    ATy = A.T @ y
    lam_max = float(np.linalg.eigvalsh(ATA)[-1])
    eta = 0.25 / lam_max

    x = nnls_solve(A, y) if x_init is None else x_init.copy()

    samples = np.empty((N, n), dtype=float)
    collected = 0
    if include_init:
        samples[0] = x
        collected = 1

    remaining = max(N - collected, 1)
    n_warmup = int(n_steps * warmup_frac)
    thin = max(1, (n_steps - n_warmup) // remaining)
    noise_scale = np.sqrt(2.0 * eta * tau)

    for t in range(n_steps):
        grad = 2.0 * (ATA @ x - ATy)
        x = np.maximum(0.0, x - eta * grad + noise_scale * rng.standard_normal(n))
        if t >= n_warmup and (t - n_warmup) % thin == 0 and collected < N:
            samples[collected] = x
            collected += 1
    return samples[:collected]
