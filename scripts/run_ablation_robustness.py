from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matchmaking_sim.experiments import aggregate_with_ci, simulate_condition


SEEDS = list(range(201, 211))


def _partial_path(out_dir: Path, scenario: str, seed: int) -> Path:
    safe = scenario.replace("=", "-").replace(";", "_")
    return out_dir / f"summary_seed{seed}_{safe}.csv"


def _run_one(args: tuple) -> tuple[str, dict | None, str]:
    out_dir, profile, seed, scenario, n_players, days, matches_per_day, overrides = args
    out_dir = Path(out_dir)
    path = _partial_path(out_dir, scenario, seed)
    if path.exists():
        try:
            return "skip", pd.read_csv(path).iloc[0].to_dict(), f"skip seed={seed} scenario={scenario}"
        except Exception:
            path.unlink(missing_ok=True)
    try:
        summary, _ = simulate_condition(
            seed=seed,
            profile=profile,
            rating_name="dynamic_k",
            policy="multi_objective",
            scenario=scenario,
            n_players=n_players,
            days=days,
            matches_per_day=matches_per_day,
            overrides=overrides,
        )
        pd.DataFrame([summary]).to_csv(path, index=False)
        return "ok", summary, f"ok seed={seed} scenario={scenario} matches={summary.get('n_matches')}"
    except Exception as exc:
        return "failed", None, f"failed seed={seed} scenario={scenario} reason={exc!r}"


def _run_suite(name: str, conditions: list[dict], n_players: int, days: int, matches_per_day: int) -> pd.DataFrame:
    out_dir = ROOT / "results/experiment_runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "results/logs" / f"{name}.log"
    workers = int(os.environ.get("MATCHMAKING_FULL_WORKERS", max(1, min(8, (os.cpu_count() or 2) - 1))))
    args = []
    for condition in conditions:
        for seed in SEEDS:
            args.append((str(out_dir), name, seed, condition["scenario"], n_players, days, matches_per_day, condition.get("overrides")))
    rows = []
    log_lines = [f"suite={name}", f"workers={workers}", f"conditions={len(conditions)}", f"seeds={SEEDS}"]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_run_one, arg): arg for arg in args}
        for i, future in enumerate(as_completed(future_map), start=1):
            status, row, msg = future.result()
            if row is not None:
                rows.append(row)
            log_lines.append(f"{status} {i}/{len(args)} {msg}")
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["scenario", "seed"]).reset_index(drop=True)
    out = ROOT / "results/tables" / f"{name}.csv"
    df.to_csv(out, index=False)
    aggregate_with_ci(df, ["scenario"]).to_csv(ROOT / "results/tables" / f"{name}_summary.csv", index=False)
    return df


def run_ablation_study() -> pd.DataFrame:
    conditions = [
        {"scenario": "ablation_default", "overrides": None},
        {"scenario": "ablation_no_role_penalty", "overrides": {"matchmaking_config": {"multi_objective_weights": {"role_penalty": 0.0}}}},
        {"scenario": "ablation_no_uncertainty_penalty", "overrides": {"matchmaking_config": {"multi_objective_weights": {"uncertainty_penalty": 0.0}}}},
        {"scenario": "ablation_no_new_player_penalty", "overrides": {"matchmaking_config": {"multi_objective_weights": {"new_player_exposure_penalty": 0.0}}}},
        {"scenario": "ablation_visible_rank_unsmoothed", "overrides": {"rating_config": {"rating": {"visible_rank_smoothing": 1.0}}}},
        {"scenario": "ablation_no_smurf_like_players", "overrides": {"player_population_config": {"population": {"smurf_ratio": 0.0}}}},
        {
            "scenario": "ablation_no_skill_drift",
            "overrides": {
                "simulation_config": {
                    "simulation": {"skill_drift_sd": 0.0, "improving_drift": 0.0, "declining_drift": 0.0, "unstable_drift_sd": 0.0},
                    "disturbances": {"skill_drift": False},
                }
            },
        },
    ]
    return _run_suite("ablation_study", conditions, n_players=3000, days=90, matches_per_day=900)


def run_population_robustness() -> pd.DataFrame:
    conditions = [
        {"scenario": "robust_population_1800", "overrides": None, "n_players": 1800},
        {"scenario": "robust_population_3000", "overrides": None, "n_players": 3000},
        {"scenario": "robust_smurf_high", "overrides": {"player_population_config": {"population": {"smurf_ratio": 0.14}}}},
        {"scenario": "robust_new_player_high", "overrides": {"player_population_config": {"population": {"new_player_ratio": 0.22}}}},
        {"scenario": "robust_role_scarcity_high", "overrides": {"simulation_config": {"simulation": {"role_scarcity_level": 0.55}}}},
        {"scenario": "robust_visible_smoothing_slow", "overrides": {"rating_config": {"rating": {"visible_rank_smoothing": 0.08}}}},
        {"scenario": "robust_activity_skew", "overrides": {"player_population_config": {"population": {"activity_rate_sd": 0.34, "low_activity_ratio": 0.22}}}},
        {"scenario": "robust_candidate_pool_small", "overrides": {"matchmaking_config": {"matchmaking": {"candidate_pool_size": 16}}}},
    ]
    rows = []
    for condition in conditions:
        n_players = int(condition.pop("n_players", 2200))
        rows.append(_run_suite(f"robustness_{condition['scenario']}", [condition], n_players=n_players, days=60, matches_per_day=600))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(ROOT / "results/tables/population_robustness_results.csv", index=False)
    aggregate_with_ci(df, ["scenario"]).to_csv(ROOT / "results/tables/population_robustness_summary.csv", index=False)
    return df


if __name__ == "__main__":
    ablation = run_ablation_study()
    robust = run_population_robustness()
    print(f"ablation rows={len(ablation)}")
    print(f"robustness rows={len(robust)}")
