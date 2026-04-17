"""PLD-initialized TuRBO, matching BO4Mob's GP / acquisition configuration.

This is a cleaned port of run115 from the research repo. The only change vs.
BO4Mob's plain TuRBO is the initialization phase: instead of Sobol quasi-random
points we evaluate PLD posterior samples. Everything else (GP kernel, trust
region, Thompson sampling with perturbation mask, success/failure thresholds)
matches BO4Mob.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Sequence, Tuple

import numpy as np
import torch

from .nnls import nnls_solve
from .pld import projected_langevin

_DTYPE = torch.double


def _fit_gp(X_norm: torch.Tensor, Y: torch.Tensor) -> "SingleTaskGP":  # noqa: F821
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.constraints import Interval
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood

    dim = X_norm.shape[-1]
    covar = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=dim,
            lengthscale_constraint=Interval(0.005, 4.0),
        )
    )
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    model = SingleTaskGP(
        X_norm,
        Y,
        covar_module=covar,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except Exception:  # noqa: BLE001 — BoTorch has many failure modes; fall back to Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            out = model(X_norm)
            loss = -mll(out, Y.flatten())
            loss.backward()
            optimizer.step()
    return model


def _turbo_candidates(
    model: "SingleTaskGP",  # noqa: F821
    x_center: torch.Tensor,
    length: float,
    n: int,
    n_cands: int,
    seed: int,
) -> torch.Tensor:
    weights = model.covar_module.base_kernel.lengthscale.detach().view(-1)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * length / 2.0, 0.0, 1.0)

    sobol = torch.quasirandom.SobolEngine(n, scramble=True, seed=seed)
    perturb = tr_lb + (tr_ub - tr_lb) * sobol.draw(n_cands).to(dtype=_DTYPE)

    prob_perturb = min(20.0 / n, 1.0)
    mask = torch.rand(n_cands, n, dtype=_DTYPE) <= prob_perturb
    empty = torch.where(mask.sum(dim=1) == 0)[0]
    if len(empty) > 0:
        mask[empty, torch.randint(0, max(n - 1, 1), size=(len(empty),))] = 1
    X_cand = x_center.expand(n_cands, n).clone()
    X_cand[mask] = perturb[mask]
    return X_cand


def pld_initialized_turbo(
    A: np.ndarray,
    y: np.ndarray,
    evaluator: Callable[[np.ndarray, str], float],
    *,
    n_init: int,
    n_epoch: int,
    batch_size: int,
    lb: float,
    ub: float,
    seed: int = 42,
    tau: float = 2.0,
    log_every: int = 20,
    on_eval: Callable[[int, str, float, float], None] | None = None,
) -> Tuple[float, dict]:
    """Run PLD+TuRBO and return (best_nrmse, diagnostics).

    evaluator(x, label) -> NRMSE (None means failed; treated as 10.0).
    on_eval(step, phase, nrmse, best_so_far) is called after every SUMO evaluation
    if provided. `phase` is "pld_init" during warm-start and "turbo" during the
    BO loop. Everything else mirrors BO4Mob's TuRBO state machine.
    """
    from botorch.generation import MaxPosteriorSampling

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    m, n = A.shape

    # Phase 1: PLD initialization
    print(f"[turbo] Phase 1: drawing {n_init} PLD samples...", flush=True)
    t0 = time.time()
    x0 = nnls_solve(A, y)
    samples = projected_langevin(A, y, tau=tau, N=n_init, seed=seed, x_init=x0)

    X_list: list[np.ndarray] = []
    Y_list: list[float] = []
    best_nrmse = float("inf")
    for i in range(n_init):
        s = np.clip(samples[i], lb, ub)
        nrmse_i = evaluator(s, f"pld_{i}")
        if nrmse_i is None:
            nrmse_i = 10.0
        X_list.append(s)
        Y_list.append(-nrmse_i)
        best_nrmse = min(best_nrmse, nrmse_i)
        if on_eval is not None:
            on_eval(i, "pld_init", float(nrmse_i), float(best_nrmse))
        if (i + 1) % 5 == 0:
            print(f"[turbo] PLD {i+1}/{n_init}: best={best_nrmse:.4f}", flush=True)
    phase1_time = time.time() - t0
    print(f"[turbo] Phase 1 done: best={best_nrmse:.4f} ({phase1_time:.0f}s)", flush=True)

    bounds_t = torch.tensor([[lb] * n, [ub] * n], dtype=_DTYPE)
    X = torch.tensor(np.array(X_list), dtype=_DTYPE)
    X_norm = (X - bounds_t[0]) / (bounds_t[1] - bounds_t[0])
    Y = torch.tensor(Y_list, dtype=_DTYPE).unsqueeze(-1)

    # Phase 2: TuRBO with PLD warm-started data
    print(f"[turbo] Phase 2: TuRBO {n_epoch} epochs x {batch_size} batch", flush=True)
    t0 = time.time()
    length = 0.8
    length_min = 0.5 ** 7
    length_max = 1.6
    sc = fc = 0
    success_tolerance = 3
    failure_tolerance = math.ceil(max(4.0 / batch_size, float(n) / batch_size))

    eval_count = 0
    for ep in range(n_epoch):
        x_center = X_norm[Y.argmax()]
        model = _fit_gp(X_norm, Y)

        n_cands = min(5000, max(2000, 200 * n))
        X_cand = _turbo_candidates(model, x_center, length, n, n_cands, seed=seed + ep)
        try:
            ts = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_new_norm = ts(X_cand, num_samples=batch_size)
        except Exception:  # noqa: BLE001
            idx = torch.randperm(n_cands)[:batch_size]
            X_new_norm = X_cand[idx]

        X_new_real = X_new_norm * (bounds_t[1] - bounds_t[0]) + bounds_t[0]
        new_Y: list[float] = []
        for ci in range(len(X_new_real)):
            x_cand = X_new_real[ci].numpy()
            nrmse_i = evaluator(x_cand, f"turbo_{eval_count}")
            eval_count += 1
            if nrmse_i is None:
                nrmse_i = 10.0
            new_Y.append(-nrmse_i)
            if nrmse_i < best_nrmse:
                best_nrmse = nrmse_i
            if on_eval is not None:
                on_eval(n_init + eval_count - 1, "turbo", float(nrmse_i), float(best_nrmse))

        new_Y_t = torch.tensor(new_Y, dtype=_DTYPE).unsqueeze(-1)
        prev_best_Y = Y.max().item()
        X_norm = torch.cat([X_norm, X_new_norm])
        Y = torch.cat([Y, new_Y_t])

        improved = new_Y_t.max().item() > prev_best_Y + 1e-3 * abs(prev_best_Y)
        if improved:
            sc += 1
            fc = 0
        else:
            fc += 1
            sc = 0
        if sc >= success_tolerance:
            length = min(2.0 * length, length_max)
            sc = 0
        elif fc >= failure_tolerance:
            length /= 2.0
            fc = 0
        if length < length_min:
            length = 0.8
            sc = fc = 0

        if (ep + 1) % log_every == 0:
            total = n_init + eval_count
            print(
                f"[turbo] Epoch {ep+1}/{n_epoch}: best={best_nrmse:.4f}, "
                f"evals={total}, TR={length:.3f}",
                flush=True,
            )

    phase2_time = time.time() - t0
    diagnostics = {
        "phase1_time_s": round(phase1_time, 1),
        "phase2_time_s": round(phase2_time, 1),
        "n_init": n_init,
        "n_epoch": n_epoch,
        "batch_size": batch_size,
        "total_evals": n_init + eval_count,
    }
    return best_nrmse, diagnostics
