#!/usr/bin/env python3
"""Run PLD-initialized TuRBO on one network against real SUMO.

Writes `results/pldturbo_<network>_seed<seed>.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

from odspld.networks import NETWORK_CONFIGS, build_network
from odspld.paths import require_sumo
from odspld.sumo import evaluate_od
from odspld.tracking import finish as wandb_finish
from odspld.tracking import init as wandb_init
from odspld.tracking import log as wandb_log
from odspld.turbo import pld_initialized_turbo


def main() -> None:
    ap = argparse.ArgumentParser(description="PLD+TuRBO with BO4Mob-style GP / acquisition.")
    ap.add_argument("--network", required=True, choices=list(NETWORK_CONFIGS))
    ap.add_argument("--date", default="221014")
    ap.add_argument("--hour", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--wandb-project", default="ODS_PLD")
    args = ap.parse_args()

    require_sumo()
    cfg = NETWORK_CONFIGS[args.network]
    hour = args.hour or cfg["default_hour"]

    A, gt_od, y, od_pairs = build_network(args.network, date=args.date, hour=hour)

    wandb_init(
        project=args.wandb_project,
        name=f"pldturbo_{args.network}_seed{args.seed}",
        group=f"pldturbo_{args.network}",
        tags=["pld_turbo", args.network, f"seed{args.seed}"],
        config={
            "method": "pld_turbo",
            "network": args.network,
            "seed": args.seed,
            "date": args.date,
            "hour": hour,
            "n": int(A.shape[1]),
            "m": int(A.shape[0]),
            "n_init": cfg["n_init_search"],
            "n_epoch": cfg["n_epoch"],
            "batch_size": cfg["bo_batch_size"],
        },
    )

    evaluator = partial(
        _evaluate,
        network=args.network,
        od_pairs=od_pairs,
        date=args.date,
        hour=hour,
        seed=args.seed,
    )

    def _on_eval(step: int, phase: str, nrmse: float, best: float) -> None:
        wandb_log(
            {
                "nrmse": nrmse,
                "best_so_far": best,
                "phase_is_turbo": 1.0 if phase == "turbo" else 0.0,
            },
            step=step,
        )

    t0 = time.time()
    best_nrmse, diag = pld_initialized_turbo(
        A, y, evaluator,
        n_init=cfg["n_init_search"],
        n_epoch=cfg["n_epoch"],
        batch_size=cfg["bo_batch_size"],
        lb=cfg["od_bound_start"],
        ub=cfg["od_bound_end"],
        seed=args.seed,
        on_eval=_on_eval,
    )
    wall = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pldturbo_{args.network}_seed{args.seed}.json"
    out_path.write_text(json.dumps({
        "method": "pld_turbo",
        "network": args.network,
        "seed": args.seed,
        "date": args.date,
        "hour": hour,
        "n": int(A.shape[1]),
        "m": int(A.shape[0]),
        "nrmse_sumo_best": best_nrmse,
        "wall_time_s": round(wall, 1),
        **diag,
    }, indent=2))
    print(f"[pldturbo/{args.network}] best NRMSE = {best_nrmse}; wrote {out_path}")

    wandb_log(
        {
            "best_nrmse_final": float(best_nrmse),
            "wall_time_s": wall,
            "total_evals": float(diag["total_evals"]),
        }
    )
    wandb_finish()


def _evaluate(x, label, *, network, od_pairs, date, hour, seed):
    return evaluate_od(
        network, x, od_pairs, date=date, hour=hour, label=f"seed{seed}_{label}"
    )


if __name__ == "__main__":
    main()
