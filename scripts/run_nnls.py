#!/usr/bin/env python3
"""Compute the NNLS solution for one network and validate it in SUMO.

Writes `results/nnls_<network>_seed<seed>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from odspld.networks import NETWORK_CONFIGS, build_network
from odspld.nnls import kkt_residual, nnls_solve
from odspld.paths import require_sumo
from odspld.sumo import evaluate_od
from odspld.tracking import finish as wandb_finish
from odspld.tracking import init as wandb_init
from odspld.tracking import log as wandb_log


def main() -> None:
    ap = argparse.ArgumentParser(description="NNLS + SUMO validation for one network.")
    ap.add_argument("--network", required=True, choices=list(NETWORK_CONFIGS))
    ap.add_argument("--date", default="221014")
    ap.add_argument("--hour", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--wandb-project", default="ODS_PLD")
    args = ap.parse_args()

    require_sumo()
    A, gt_od, y, od_pairs = build_network(args.network, date=args.date, hour=args.hour)
    m, n = A.shape

    wandb_init(
        project=args.wandb_project,
        name=f"nnls_{args.network}_seed{args.seed}",
        group=f"nnls_{args.network}",
        tags=["nnls", args.network, f"seed{args.seed}"],
        config={
            "method": "nnls",
            "network": args.network,
            "seed": args.seed,
            "date": args.date,
            "hour": args.hour or NETWORK_CONFIGS[args.network]["default_hour"],
            "n": int(n),
            "m": int(m),
        },
    )

    t0 = time.time()
    x = nnls_solve(A, y)
    nnls_time = time.time() - t0
    kkt = kkt_residual(A, y, x)

    print(f"[nnls/{args.network}] A={m}x{n}; ||A x - y||^2 = {(A @ x - y) @ (A @ x - y):.4f}")
    print(f"[nnls/{args.network}] KKT residual = {kkt}")
    print(f"[nnls/{args.network}] running SUMO validation...", flush=True)

    nrmse = evaluate_od(
        args.network,
        x,
        od_pairs,
        date=args.date,
        hour=args.hour or NETWORK_CONFIGS[args.network]["default_hour"],
        label=f"nnls_seed{args.seed}",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "method": "nnls",
        "network": args.network,
        "seed": args.seed,
        "date": args.date,
        "hour": args.hour or NETWORK_CONFIGS[args.network]["default_hour"],
        "n": int(n),
        "m": int(m),
        "nrmse_sumo": nrmse,
        "kkt": kkt,
        "time_s": round(nnls_time, 4),
    }
    out_path = out_dir / f"nnls_{args.network}_seed{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[nnls/{args.network}] SUMO NRMSE = {nrmse}; wrote {out_path}")

    wandb_log(
        {
            "nrmse_sumo": float(nrmse) if nrmse is not None else float("nan"),
            "kkt_primal_min": kkt["primal_min"],
            "kkt_dual_min": kkt["dual_min"],
            "kkt_complementary_slackness": kkt["complementary_slackness"],
            "time_s": nnls_time,
        }
    )
    wandb_finish()


if __name__ == "__main__":
    main()
