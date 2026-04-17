"""Evaluate an OD vector with SUMO using BO4Mob's `single_od_run.py`.

Writes a temporary OD CSV under `od_for_single_run/`, runs SUMO, and parses the
NRMSE out of `Loss: <value>`. Cleans up both the temp CSV and SUMO's per-run
output directory to avoid cross-run contamination.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import uuid
from typing import Sequence, Tuple

from .paths import bo4mob_path


def _write_temp_od(
    network: str,
    od_values: Sequence[float],
    od_pairs: Sequence[Tuple[str, str]],
    bo4mob: str,
    label: str,
) -> str:
    src = os.path.join(bo4mob, "od_for_single_run", f"od_{network}.csv")
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))

    od_dict = dict(zip(od_pairs, od_values))
    tmp_name = f"od_{network}_{label}.csv"
    tmp_path = os.path.join(bo4mob, "od_for_single_run", tmp_name)
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            key = (row["fromTaz"].strip(), row["toTaz"].strip())
            value = od_dict.get(key, float(row["flow"]))
            out = dict(row)
            out["flow"] = str(max(0.0, round(float(value), 1)))
            writer.writerow(out)
    return tmp_name


def evaluate_od(
    network: str,
    od_values: Sequence[float],
    od_pairs: Sequence[Tuple[str, str]],
    date: str = "221014",
    hour: str = "08-09",
    label: str | None = None,
    timeout: int = 600,
    bo4mob: str | None = None,
) -> float | None:
    """Run SUMO for `od_values` on `network` and return the NRMSE loss."""
    bo4mob = bo4mob or bo4mob_path()
    label = label or uuid.uuid4().hex[:8]
    tmp_csv = _write_temp_od(network, od_values, od_pairs, bo4mob, label)
    tmp_abs = os.path.join(bo4mob, "od_for_single_run", tmp_csv)

    out_dir = os.path.join(
        bo4mob,
        "output",
        "single_od_run",
        f"network_{network}_{date}_{hour}_count_single_{tmp_csv[:-4]}_csv",
    )
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    cmd = [
        "python3",
        os.path.join(bo4mob, "src", "single_od_run.py"),
        "--network_name", network,
        "--date", date,
        "--hour", hour,
        "--eval_measure", "count",
        "--routes_per_od", "single",
        "--od_csv", tmp_csv,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=bo4mob,
            timeout=timeout,
            check=False,
        )
    finally:
        if os.path.exists(tmp_abs):
            os.remove(tmp_abs)

    nrmse: float | None = None
    for line in result.stdout.splitlines():
        if "Loss:" in line:
            try:
                nrmse = float(line.split("Loss:")[1].strip())
            except ValueError:
                nrmse = None
    if nrmse is None:
        import sys as _sys
        print(
            f"[evaluate_od] SUMO returned no 'Loss:' line for {network}/{label}. "
            f"returncode={result.returncode}",
            file=_sys.stderr,
            flush=True,
        )
        if result.stderr:
            print(f"[evaluate_od] stderr:\n{result.stderr[-4000:]}", file=_sys.stderr, flush=True)
        if result.stdout:
            print(f"[evaluate_od] stdout tail:\n{result.stdout[-2000:]}", file=_sys.stderr, flush=True)
    return nrmse
