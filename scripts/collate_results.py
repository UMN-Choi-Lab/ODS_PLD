#!/usr/bin/env python3
"""Collect all JSON results under `results/` and print a LaTeX / markdown table.

Usage:
    python scripts/collate_results.py --format latex   > table.tex
    python scripts/collate_results.py --format md     > table.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METHOD_LABELS = [
    ("nnls", "NNLS"),
    ("pld_bestofN", "PLD best-N"),
    ("pld_turbo", "PLD+TuRBO"),
]

NETWORKS = ["1ramp", "2corridor", "3junction", "4smallRegion"]


def _load_results(results_dir: Path) -> dict:
    """Return {network: {method: [nrmse_per_seed, ...]}}."""
    agg: dict = {net: {m: [] for m, _ in METHOD_LABELS} for net in NETWORKS}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        net = data.get("network")
        method = data.get("method")
        if net not in agg or method not in agg[net]:
            continue
        if method == "pld_bestofN":
            nrmse = data.get("nrmse_sumo_best")
        elif method == "pld_turbo":
            nrmse = data.get("nrmse_sumo_best")
        else:
            nrmse = data.get("nrmse_sumo")
        if nrmse is not None:
            agg[net][method].append(float(nrmse))
    return agg


def _cell(values: list[float]) -> str:
    if not values:
        return "---"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f}"


def _latex(agg: dict) -> str:
    header = " & ".join(["Network"] + [lbl for _, lbl in METHOD_LABELS]) + r" \\"
    rows = [r"\begin{tabular}{l" + "c" * len(METHOD_LABELS) + "}", r"\toprule", header, r"\midrule"]
    for net in NETWORKS:
        cells = [_cell(agg[net][m]) for m, _ in METHOD_LABELS]
        rows.append(f"{net} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def _markdown(agg: dict) -> str:
    header = "| Network | " + " | ".join(lbl for _, lbl in METHOD_LABELS) + " |"
    sep = "|" + "|".join(["---"] * (len(METHOD_LABELS) + 1)) + "|"
    lines = [header, sep]
    for net in NETWORKS:
        cells = [_cell(agg[net][m]) for m, _ in METHOD_LABELS]
        lines.append(f"| {net} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--format", choices=["latex", "md"], default="md")
    args = ap.parse_args()

    agg = _load_results(Path(args.results_dir))
    if args.format == "latex":
        print(_latex(agg))
    else:
        print(_markdown(agg))


if __name__ == "__main__":
    main()
