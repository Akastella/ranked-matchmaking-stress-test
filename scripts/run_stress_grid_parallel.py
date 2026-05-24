from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from parallel_utils import default_n_jobs, run_parallel_tasks
from matchmaking_sim.experiments import simulate_condition


def _task(task: dict) -> dict:
    smoothing = {"fast": 0.35, "default": 0.22, "slow": 0.10}[task["visible_rank_smoothing"]]
    scarcity = {"low": 0.08, "high": 0.45}[task["role_scarcity"]]
    pool = {"small": 16, "large": 42}[task["candidate_pool_size"]]
    weights = {
        "default": {"fairness_loss": 0.70, "waiting_cost": 0.18, "role_penalty": 0.35, "uncertainty_penalty": 0.10, "new_player_exposure_penalty": 0.30},
        "subgroup_heavy": {"fairness_loss": 0.60, "waiting_cost": 0.16, "role_penalty": 0.35, "uncertainty_penalty": 0.12, "new_player_exposure_penalty": 0.45, "smurf_exposure_penalty": 0.45},
    }[task["weight_set"]]
    policy = "multi_objective_with_smurf_exposure_penalty" if task["weight_set"] == "subgroup_heavy" else "multi_objective"
    overrides = {
        "player_population_config": {"population": {"smurf_ratio": task["smurf_ratio"]}},
        "simulation_config": {"simulation": {"role_scarcity_level": scarcity}},
        "rating_config": {"rating": {"visible_rank_smoothing": smoothing}},
        "matchmaking_config": {"matchmaking": {"candidate_pool_size": pool}, "multi_objective_weights": weights},
    }
    summary, _ = simulate_condition(
        int(task["seed"]),
        "stress_grid_compact",
        "dynamic_k",
        policy,
        task["scenario"],
        n_players=1100,
        days=22,
        matches_per_day=180,
        overrides=overrides,
    )
    summary.update(task)
    return summary


def build_tasks() -> list[dict]:
    tasks = []
    for seed in [801, 802, 803, 804]:
        for smurf_ratio in [0.0, 0.05, 0.10]:
            for role in ["low", "high"]:
                for smoothing in ["fast", "slow"]:
                    for pool in ["small", "large"]:
                        for weight_set in ["default", "subgroup_heavy"]:
                            tasks.append(
                                {
                                    "task_id": f"grid_seed{seed}_sr{smurf_ratio}_{role}_{smoothing}_{pool}_{weight_set}",
                                    "seed": seed,
                                    "scenario": "stress_grid",
                                    "smurf_ratio": smurf_ratio,
                                    "role_scarcity": role,
                                    "visible_rank_smoothing": smoothing,
                                    "candidate_pool_size": pool,
                                    "weight_set": weight_set,
                                }
                            )
    return tasks


if __name__ == "__main__":
    df, stats = run_parallel_tasks(
        build_tasks(),
        _task,
        ROOT / "results/experiment_runs/stress_grid_parallel",
        ROOT / "results/logs/parallel_run_stress_grid.log",
        default_n_jobs(),
    )
    out = ROOT / "results/tables/stress_grid_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"stress_grid rows={len(df)} stats={stats}")
