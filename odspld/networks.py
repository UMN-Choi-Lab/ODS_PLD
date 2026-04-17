"""Build assignment matrices, ground-truth ODs, and sensor counts from BO4Mob.

BO4Mob is vendored as a git submodule under `external/BO4Mob`. Each network's
assignment matrix is built from `routes_single.csv` (all-or-nothing single
shortest path per OD) and sensor CSVs under `sensor_data/<date>/`.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Tuple

import numpy as np

from .paths import bo4mob_path


NETWORK_CONFIGS = {
    "1ramp": {
        "default_hour": "08-09",
        "od_bound_start": 1.0,
        "od_bound_end": 2500.0,
        "n_init_search": 10,
        "n_epoch": 50,
        "bo_batch_size": 2,
    },
    "2corridor": {
        "default_hour": "08-09",
        "od_bound_start": 1.0,
        "od_bound_end": 2000.0,
        "n_init_search": 20,
        "n_epoch": 100,
        "bo_batch_size": 3,
    },
    "3junction": {
        "default_hour": "08-09",
        "od_bound_start": 1.0,
        "od_bound_end": 2000.0,
        "n_init_search": 30,
        "n_epoch": 200,
        "bo_batch_size": 4,
    },
    "4smallRegion": {
        "default_hour": "17-18",
        "od_bound_start": 1.0,
        "od_bound_end": 2000.0,
        "n_init_search": 50,
        "n_epoch": 600,
        "bo_batch_size": 5,
    },
}


def _read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_routes(network: str, bo4mob: str) -> list[Tuple[str, str, list[str], float]]:
    path = os.path.join(bo4mob, "network", f"network_{network}", "routes_single.csv")
    out = []
    for row in _read_csv_rows(path):
        edges = row["route_edges"].strip().split()
        ratio = float(row.get("ratio", 1.0))
        out.append((row["fromTaz"].strip(), row["toTaz"].strip(), edges, ratio))
    return out


def _read_sensor_data(network: str, date: str, hour: str, bo4mob: str) -> Tuple[list[str], np.ndarray]:
    path = os.path.join(
        bo4mob, "sensor_data", date, f"gt_link_data_{network}_{date}_{hour}.csv"
    )
    rows = _read_csv_rows(path)
    link_ids = [r["link_id"].strip() for r in rows]
    counts = np.array([float(r["interval_nVehContrib"]) for r in rows])
    return link_ids, counts


def _read_ground_truth_od(network: str, bo4mob: str) -> list[Tuple[Tuple[str, str], float]]:
    path = os.path.join(bo4mob, "od_for_single_run", f"od_{network}.csv")
    out = []
    for row in _read_csv_rows(path):
        key = (row["fromTaz"].strip(), row["toTaz"].strip())
        out.append((key, float(row["flow"])))
    return out


def _build_A(
    od_order: Iterable[Tuple[str, str]],
    od_routes: dict[Tuple[str, str], list[Tuple[float, list[str]]]],
    sensor_links: list[str],
) -> np.ndarray:
    od_order = list(od_order)
    m = len(sensor_links)
    n = len(od_order)
    A = np.zeros((m, n))
    sensor_index = {link: i for i, link in enumerate(sensor_links)}
    for j, key in enumerate(od_order):
        for ratio, edges in od_routes.get(key, []):
            for edge in edges:
                i = sensor_index.get(edge)
                if i is not None:
                    A[i, j] += ratio
    return A


def build_network(
    network: str,
    date: str = "221014",
    hour: str | None = None,
    bo4mob: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[Tuple[str, str]]]:
    """Build (A, gt_od, counts, od_pairs) for a BO4Mob network.

    od_pairs is ordered to match the columns of A and the rows of gt_od and
    follows the canonical order defined by `od_<network>.csv` (the same order
    SUMO expects when writing back).
    """
    if network not in NETWORK_CONFIGS:
        raise KeyError(f"Unknown network {network!r}. Choose from {list(NETWORK_CONFIGS)}.")
    cfg = NETWORK_CONFIGS[network]
    hour = hour or cfg["default_hour"]
    bo4mob = bo4mob or bo4mob_path()

    routes = _read_routes(network, bo4mob)
    od_routes: dict[Tuple[str, str], list[Tuple[float, list[str]]]] = {}
    for from_taz, to_taz, edges, ratio in routes:
        od_routes.setdefault((from_taz, to_taz), []).append((ratio, edges))

    gt_pairs = _read_ground_truth_od(network, bo4mob)
    od_order = [pair for pair, _ in gt_pairs]
    gt_od = np.array([flow for _, flow in gt_pairs])

    sensor_links, counts = _read_sensor_data(network, date, hour, bo4mob)
    A = _build_A(od_order, od_routes, sensor_links)

    return A, gt_od, counts, od_order
