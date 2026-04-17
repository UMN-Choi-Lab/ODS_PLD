"""Resolve paths to the BO4Mob submodule and SUMO installation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bo4mob_path() -> str:
    env = os.environ.get("BO4MOB_PATH")
    if env:
        return env
    default = repo_root() / "external" / "BO4Mob"
    if not default.exists():
        raise FileNotFoundError(
            f"BO4Mob not found at {default}. Initialize the submodule with:\n"
            "    git submodule update --init --recursive\n"
            "or set BO4MOB_PATH in your environment."
        )
    return str(default)


def sumo_home() -> str:
    env = os.environ.get("SUMO_HOME")
    if env:
        return env
    for candidate in ("/usr/share/sumo", "/usr/local/share/sumo"):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "SUMO not found. Install with `apt-get install sumo sumo-tools` "
        "(Ubuntu) and export SUMO_HOME=/usr/share/sumo."
    )


def require_sumo() -> None:
    sumo_home()
    if shutil.which("sumo") is None:
        raise RuntimeError("The `sumo` binary is not on PATH.")
