#!/usr/bin/env python3
"""Draw N PLD samples, evaluate each in SUMO, and keep the best.

Writes `results/pld_<network>_seed<seed>.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from odspld.networks import NETWORK_CONFIGS, build_network
from odspld.paths import require_sumo
from odspld.pld import projected_langevin
from odspld.sumo import evaluate_od
from odspld.tracking import finish as wandb_finish
from odspld.tracking import init as wandb_init
from odspld.tracking import log as wandb_log


def main() -> None:
    ap = argparse.ArgumentParser(description="PLD best-of-N under SUMO validation.")
    ap.add_argument("--network", required=True, choices=list(NETWORK_CONFIGS))
    ap.add_argument("--date", default="221014")
    ap.add_argument("--hour", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--N", type=int, default=20,
        help="Total candidates to evaluate in SUMO (sample 1 is NNLS; samples 2..N are PLD draws).",
    )
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--wandb-project", default="ODS_PLD")
    args = ap.parse_args()

    require_sumo()
    A, gt_od, y, od_pairs = build_network(args.network, date=args.date, hour=args.hour)
    hour = args.hour or NETWORK_CONFIGS[args.network]["default_hour"]

    wandb_init(
        project=args.wandb_project,
        name=f"pld_{args.network}_seed{args.seed}",
        group=f"pld_{args.network}",
        tags=["pld_bestofN", args.network, f"seed{args.seed}"],
        config={
            "method": "pld_bestofN",
            "network": args.network,
            "seed": args.seed,
            "date": args.date,
            "hour": hour,
            "N": args.N,
            "tau": args.tau,
        },
    )

    t0 = time.time()
    samples = projected_langevin(A, y, tau=args.tau, N=args.N, seed=args.seed)
    pld_time = time.time() - t0

    best_nrmse = float("inf")
    per_sample = []
    for i, x in enumerate(samples):
        nrmse = evaluate_od(args.network, x, od_pairs, date=args.date, hour=hour,
                            label=f"pld_seed{args.seed}_{i}")
        per_sample.append(nrmse)
        if nrmse is not None and nrmse < best_nrmse:
            best_nrmse = nrmse
        print(f"[pld/{args.network}] sample {i+1}/{args.N}: {nrmse}", flush=True)
        wandb_log(
            {
                "sample_nrmse": float(nrmse) if nrmse is not None else float("nan"),
                "best_so_far": float(best_nrmse) if best_nrmse < float("inf") else float("nan"),
            },
            step=i,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pld_{args.network}_seed{args.seed}.json"
    out_path.write_text(json.dumps({
        "method": "pld_bestofN",
        "network": args.network,
        "seed": args.seed,
        "date": args.date,
        "hour": hour,
        "N": args.N,
        "tau": args.tau,
        "nrmse_sumo_best": best_nrmse if best_nrmse < float("inf") else None,
        "nrmse_sumo_per_sample": per_sample,
        "pld_time_s": round(pld_time, 1),
    }, indent=2))
    print(f"[pld/{args.network}] best NRMSE = {best_nrmse}; wrote {out_path}")

    wandb_log(
        {
            "best_nrmse_final": (
                float(best_nrmse) if best_nrmse < float("inf") else float("nan")
            ),
            "pld_time_s": pld_time,
        }
    )
    wandb_finish()


if __name__ == "__main__":
    main()
