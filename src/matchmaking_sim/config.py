from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all_configs(root: str | Path | None = None) -> dict[str, Any]:
    base = Path(root) if root else PROJECT_ROOT
    cfg: dict[str, Any] = {}
    for name in [
        "experiment_config",
        "player_population_config",
        "rating_config",
        "matchmaking_config",
        "simulation_config",
        "manuscript_config",
    ]:
        cfg[name] = load_yaml(base / "config" / f"{name}.yaml")
    return cfg


def ensure_dirs(root: str | Path | None = None) -> None:
    base = Path(root) if root else PROJECT_ROOT
    for rel in [
        "results/figures",
        "results/tables",
        "results/logs",
        "results/experiment_runs",
        "results/sensitivity",
        "results/diagnostics",
        "data/processed",
        "data/synthetic",
        "submission/source",
        "submission/figures",
        "submission/tables",
    ]:
        (base / rel).mkdir(parents=True, exist_ok=True)
