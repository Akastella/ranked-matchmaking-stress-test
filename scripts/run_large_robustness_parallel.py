from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from parallel_utils import default_n_jobs, run_parallel_tasks
from matchmaking_sim.experiments import simulate_condition


def _task(task: dict) -> dict:
    overrides = task.get("overrides") or {}
    summary, _ = simulate_condition(
        int(task["seed"]),
        "large_robustness_compact",
        "dynamic_k",
        task["policy"],
        task["scenario"],
        n_players=int(task["n_players"]),
        days=int(task["days"]),
        matches_per_day=int(task["matches_per_day"]),
        overrides=overrides,
    )
    summary.update({k: v for k, v in task.items() if k != "overrides"})
    return summary


def build_tasks() -> list[dict]:
    seeds = [701, 702, 703, 704, 705, 706]
    policies = ["nearest", "role_aware", "multi_objective", "multi_objective_with_smurf_exposure_penalty"]
    scenarios = {
        "baseline": {},
        "smurf_disruption": {"player_population_config": {"population": {"smurf_ratio": 0.12}}},
        "role_scarcity": {"simulation_config": {"simulation": {"role_scarcity_level": 0.45}}},
        "rank_mmr_divergence": {"rating_config": {"rating": {"visible_rank_smoothing": 0.10}}},
    }
    tasks = []
    for seed in seeds:
        for scenario, overrides in scenarios.items():
            for policy in policies:
                tasks.append(
                    {
                        "task_id": f"large_seed{seed}_{scenario}_{policy}",
                        "seed": seed,
                        "scenario": scenario,
                        "policy": policy,
                        "n_players": 1600,
                        "days": 30,
                        "matches_per_day": 240,
                        "overrides": overrides,
                    }
                )
    return tasks


if __name__ == "__main__":
    df, stats = run_parallel_tasks(
        build_tasks(),
        _task,
        ROOT / "results/experiment_runs/large_robustness_parallel",
        ROOT / "results/logs/parallel_run_large_robustness.log",
        default_n_jobs(),
    )
    out = ROOT / "results/tables/large_robustness_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"large_robustness rows={len(df)} stats={stats}")
