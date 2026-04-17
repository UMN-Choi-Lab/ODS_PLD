#!/usr/bin/env python3
"""End-to-end orchestrator: reproduce the SUMO table across networks and seeds.

Runs NNLS -> PLD best-of-N -> PLD+TuRBO for each (network, seed), writing one
JSON per (method, network, seed) into `results/`. When everything is done it
calls `collate_results.py` so the final table is printed to stdout.

Expected wall clock on a workstation (single-threaded per cell):
    1ramp:        ~10 min per seed
    2corridor:    ~1 hr per seed
    3junction:    ~6 hr per seed
    4smallRegion: ~40 hr per seed   <-- dominates
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_NETWORKS = ["1ramp", "2corridor", "3junction", "4smallRegion"]
DEFAULT_SEEDS = [11, 12, 13]
SCRIPT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=DEFAULT_NETWORKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--date", default="221014")
    ap.add_argument("--hour", default=None)
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["nnls", "pld_bestofN", "pld_turbo"],
        choices=["nnls", "pld_bestofN", "pld_turbo"],
    )
    ap.add_argument("--N", type=int, default=20, help="N for PLD best-of-N.")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs: list[list[str]] = []
    for net in args.networks:
        for seed in args.seeds:
            base = ["--network", net, "--seed", str(seed), "--date", args.date,
                    "--out-dir", args.out_dir]
            if args.hour:
                base += ["--hour", args.hour]
            if "nnls" in args.methods:
                jobs.append([sys.executable, str(SCRIPT_DIR / "run_nnls.py"), *base])
            if "pld_bestofN" in args.methods:
                jobs.append([sys.executable, str(SCRIPT_DIR / "run_pld_bestofN.py"),
                             *base, "--N", str(args.N)])
            if "pld_turbo" in args.methods:
                jobs.append([sys.executable, str(SCRIPT_DIR / "run_pld_turbo.py"), *base])

    print(f"Planned {len(jobs)} jobs.", flush=True)
    if args.dry_run:
        for cmd in jobs:
            print(" ".join(cmd))
        return

    for cmd in jobs:
        _run(cmd)

    _run([sys.executable, str(SCRIPT_DIR / "collate_results.py"),
          "--results-dir", args.out_dir, "--format", "md"])


if __name__ == "__main__":
    main()
