"""Thin W&B wrapper that no-ops outside an active run.

Usage
-----
    from odspld.tracking import init, log, finish

    init(project="ODS_PLD",
         name="nnls_2corridor_seed0",
         group="nnls_2corridor",
         tags=["nnls", "2corridor", "seed0"],
         config={"seed": 0, "network": "2corridor"})

    log({"nrmse_sumo": 0.18, "best_so_far": 0.18}, step=3)

    finish()

Inside a container with `WANDB_API_KEY` set the payload is forwarded to
`wandb.log`. Outside W&B (no key / no install / init failure) every call
becomes a no-op so scripts stay environment-agnostic.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

try:
    import wandb as _wandb  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    _wandb = None

_run = None


def _is_finite_number(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(float(v))
    return False


def init(
    *,
    project: str,
    name: str,
    group: str | None = None,
    tags: Iterable[str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Start a W&B run. No-op if wandb is unavailable or init fails."""
    global _run
    if _wandb is None:
        return
    try:
        _run = _wandb.init(
            project=project,
            name=name,
            group=group,
            tags=list(tags) if tags is not None else None,
            config=dict(config) if config is not None else None,
            reinit=False,
        )
    except Exception:  # noqa: BLE001
        _run = None


def log(payload: Mapping[str, Any], step: int | None = None) -> None:
    """Forward numeric metrics to `wandb.log`; drop None / NaN / Inf / non-numerics.

    W&B itself tolerates NaN (plots show gaps) but the VESSL legacy code had
    stricter validation; we keep the filter for cross-backend safety.
    """
    if _run is None or _wandb is None:
        return
    filtered = {k: float(v) for k, v in payload.items() if _is_finite_number(v)}
    if not filtered:
        return
    try:
        if step is None:
            _wandb.log(filtered)
        else:
            _wandb.log(filtered, step=step)
    except Exception:  # noqa: BLE001
        pass


def finish() -> None:
    """End the active W&B run (safe to call even when none is active)."""
    global _run
    if _run is None or _wandb is None:
        return
    try:
        _wandb.finish()
    except Exception:  # noqa: BLE001
        pass
    _run = None
