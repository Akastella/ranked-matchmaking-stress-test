from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from parallel_utils import default_n_jobs, run_parallel_tasks
from matchmaking_sim.experiments import simulate_condition


def _task(task: dict) -> dict:
    gap = float(task["smurf_skill_gap"])
    overrides = {
        "player_population_config": {
            "population": {
                "smurf_ratio": task["smurf_ratio"],
                "new_player_ratio": task["new_player_ratio"],
                "smurf_latent_skill_boost": gap / 400.0,
                "smurf_initial_mmr_gap_min": gap * 0.7,
                "smurf_initial_mmr_gap_max": gap * 1.15,
            }
        },
        "simulation_config": {"simulation": {"role_scarcity_level": task["role_scarcity_level"]}},
        "matchmaking_config": {
            "matchmaking": {"candidate_pool_size": task["candidate_pool_size"]},
            "multi_objective_weights": task["weights"],
        },
    }
    summary, _ = simulate_condition(
        int(task["seed"]),
        "risk_transfer",
        "dynamic_k",
        task["policy"],
        task["scenario"],
        n_players=int(task["n_players"]),
        days=int(task["days"]),
        matches_per_day=int(task["matches_per_day"]),
        overrides=overrides,
    )
    summary.update({k: v for k, v in task.items() if k not in {"weights"}})
    return summary


def build_tasks() -> list[dict]:
    seeds = [601, 602, 603, 604]
    smurf_ratios = [0.00, 0.02, 0.05, 0.10, 0.15]
    smurf_gaps = [300, 700]
    role_levels = {"low": 0.08, "high": 0.45}
    policies = ["nearest", "role_aware", "multi_objective_without_smurf_penalty", "multi_objective_with_smurf_exposure_penalty"]
    weight_sets = {
        "default": {
            "fairness_loss": 0.70,
            "waiting_cost": 0.18,
            "role_penalty": 0.35,
            "uncertainty_penalty": 0.10,
            "new_player_exposure_penalty": 0.30,
            "smurf_exposure_penalty": 0.00,
        },
        "smurf_penalty_added": {
            "fairness_loss": 0.70,
            "waiting_cost": 0.18,
            "role_penalty": 0.35,
            "uncertainty_penalty": 0.10,
            "new_player_exposure_penalty": 0.30,
            "smurf_exposure_penalty": 0.65,
        },
    }
    tasks = []
    for seed in seeds:
        for smurf_ratio in smurf_ratios:
            for gap in smurf_gaps:
                for role_label, role_level in role_levels.items():
                    for policy in policies:
                        weights = weight_sets["smurf_penalty_added" if policy.endswith("penalty") else "default"]
                        task_id = f"risk_seed{seed}_sr{smurf_ratio}_gap{gap}_{role_label}_{policy}"
                        tasks.append(
                            {
                                "task_id": task_id,
                                "seed": seed,
                                "smurf_ratio": smurf_ratio,
                                "smurf_skill_gap": gap,
                                "role_scarcity": role_label,
                                "role_scarcity_level": role_level,
                                "new_player_ratio": 0.10,
                                "candidate_pool_size": 30,
                                "policy": policy,
                                "scenario": "risk_transfer",
                                "n_players": 720,
                                "days": 16,
                                "matches_per_day": 120,
                                "weights": weights,
                            }
                        )
    return tasks


if __name__ == "__main__":
    tasks = build_tasks()
    df, stats = run_parallel_tasks(
        tasks,
        _task,
        ROOT / "results/experiment_runs/risk_transfer_parallel",
        ROOT / "results/logs/parallel_run_risk_transfer.log",
        default_n_jobs(),
    )
    out = ROOT / "results/tables/risk_transfer_experiment.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"risk_transfer rows={len(df)} stats={stats}")
