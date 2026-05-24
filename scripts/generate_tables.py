from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def write_table(name: str, df: pd.DataFrame) -> None:
    out = ROOT / "results/tables" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    df.to_latex(out.with_suffix(".tex"), index=False, escape=True)


def load_yaml(rel: str):
    with (ROOT / rel).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def first_existing(*rels: str) -> Path:
    for rel in rels:
        path = ROOT / rel
        if path.exists():
            return path
    return ROOT / rels[-1]


def bootstrap_ci(values, n_boot=2000, seed=123):
    import numpy as np
    vals = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if len(vals) == 0:
        return (float('nan'), float('nan'), float('nan'))
    boots = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    return float(np.mean(vals)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def cliffs_delta(x, y):
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    gt = sum(float(a > b) for a in x for b in y)
    lt = sum(float(a < b) for a in x for b in y)
    return float((gt - lt) / (len(x) * len(y)))


if __name__ == "__main__":
    sim = load_yaml("config/simulation_config.yaml")
    pop = load_yaml("config/player_population_config.yaml")
    rating = load_yaml("config/rating_config.yaml")
    mm = load_yaml("config/matchmaking_config.yaml")
    exp = load_yaml("config/experiment_config.yaml")

    write_table(
        "table_01_simulation_parameters.csv",
        pd.DataFrame(
            [
                {"parameter": "population_size_full_core", "value": exp["full_profile"]["n_players"], "source": "experiment_config.yaml"},
                {"parameter": "simulation_days_full_core", "value": exp["full_profile"]["days"], "source": "experiment_config.yaml"},
                {"parameter": "matches_per_day_full_core", "value": exp["full_profile"]["matches_per_day"], "source": "experiment_config.yaml"},
                {"parameter": "full_core_seeds", "value": "201-210", "source": "scripts/run_full_experiments.py"},
                {"parameter": "smurf_ratio", "value": pop["population"]["smurf_ratio"], "source": "player_population_config.yaml"},
                {"parameter": "seasonal_reset_day", "value": sim["simulation"]["seasonal_reset_day"], "source": "simulation_config.yaml"},
                {"parameter": "base_window", "value": mm["matchmaking"]["base_window"], "source": "matchmaking_config.yaml"},
                {"parameter": "churn_proxy_weights", "value": str(sim["experience_proxy"]), "source": "simulation_config.yaml"},
            ]
        ),
    )
    write_table(
        "table_02_player_state_variables.csv",
        pd.DataFrame(
            [
                {"variable": "latent_global_skill", "description": "Synthetic unobserved global skill"},
                {"variable": "latent_role_skill", "description": "Role-specific latent offsets"},
                {"variable": "hidden_mmr", "description": "Continuous matchmaking rating"},
                {"variable": "visible_rank", "description": "Smoothed user-facing rank tier"},
                {"variable": "rating_uncertainty", "description": "Update responsiveness and inactivity growth"},
                {"variable": "behavior_type", "description": "regular, new_player, improving, declining, smurf, unstable, low_activity"},
            ]
        ),
    )
    write_table(
        "table_03_rating_mechanisms.csv",
        pd.DataFrame(
            [
                {"mechanism": "Elo", "implemented_as": "constant-K hidden MMR update", "boundary": "baseline"},
                {"mechanism": "Dynamic-K Elo", "implemented_as": "K varies by uncertainty, experience, volatility", "boundary": "heuristic"},
                {"mechanism": "Glicko-like", "implemented_as": "rating deviation affects K and inactivity", "boundary": "approximation"},
                {"mechanism": "Approximate team update", "implemented_as": "team mean/variance update", "boundary": "not full factor graph"},
                {"mechanism": "Drift-aware", "implemented_as": "inactivity uncertainty and decay", "boundary": "stylized drift model"},
                {"mechanism": "Context-aware", "implemented_as": "role and context rating components", "boundary": "role component emphasized"},
            ]
        ),
    )
    write_table(
        "table_04_matchmaking_policies.csv",
        pd.DataFrame(
            [
                {"policy": p, "description": d}
                for p, d in [
                    ("random", "Random baseline"),
                    ("nearest", "Nearest hidden-MMR baseline"),
                    ("expanding", "Waiting-time expanding MMR window"),
                    ("role_aware", "Role-aware 5v5 candidate formation"),
                    ("fairness_constrained", "Expected win probability constraint"),
                    ("latency_constrained", "Waiting threshold emphasized"),
                    ("multi_objective", "Weighted fairness, waiting, role, uncertainty, and exposure score"),
                ]
            ]
        ),
    )
    main = pd.read_csv(
        first_existing(
            "results/tables/main_results_full_profile.csv",
            "results/tables/main_results_full_or_medium.csv",
            "results/tables/main_results.csv",
        )
    )
    baseline = main[main["scenario"].isin(["rating_validity", "team_matchmaking", "baseline"])].copy()
    if baseline.empty:
        baseline = main
    write_table(
        "table_05_main_experimental_results.csv",
        baseline.groupby(["rating", "policy"], as_index=False)[
            ["rating_mae", "rating_rmse", "expected_winprob_deviation", "waiting_time_proxy", "mmr_rank_divergence", "churn_risk_proxy"]
        ].mean(),
    )
    write_table(
        "table_06_tail_risk_churn_results.csv",
        main.groupby(["policy"], as_index=False)[
            [
                "tail_unfairness_p95",
                "cvar_unfairness_worst5",
                "new_player_unfair_exposure",
                "smurf_victimization",
                "mmr_rank_divergence",
                "churn_risk_proxy",
            ]
        ].mean(),
    )
    sens_path = first_existing("results/tables/proxy_weight_sensitivity_results.csv", "results/tables/sensitivity_results_full_or_medium.csv", "results/tables/sensitivity_results.csv")
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        if "smurf_ratio" in sens.columns and "drift_intensity" in sens.columns:
            group_cols = ["smurf_ratio", "drift_intensity"]
        else:
            group_cols = ["scenario"]
        write_table(
            "table_07_sensitivity_summary.csv",
            sens.groupby(group_cols, as_index=False)[
                ["expected_winprob_deviation", "rating_mae", "mmr_rank_divergence", "churn_risk_proxy"]
            ].mean(),
        )
    write_table(
        "table_11_proxy_weight_sensitivity.csv",
        main[main["scenario"].str.startswith("proxy_", na=False)]
        .groupby("scenario", as_index=False)[
            [
                "unfair_match_exposure",
                "repeated_large_gap_exposure",
                "waiting_time_frustration",
                "mmr_rank_divergence",
                "churn_risk_proxy",
            ]
        ]
        .mean(),
    )
    write_table(
        "table_12_population_grounding_notes.csv",
        pd.DataFrame(
            [
                {
                    "assumption": "Activity heterogeneity",
                    "status": "stylized",
                    "grounding": "The generator uses heterogeneous activity rates and a low-activity type; parameters are not fitted to public telemetry.",
                    "manuscript_use": "reported as an assumption and sensitivity target",
                },
                {
                    "assumption": "Ranked progression separation",
                    "status": "public-context motivated",
                    "grounding": "Riot support materials distinguish player-facing rank/RR/LP from hidden MMR or matchmaking estimates.",
                    "manuscript_use": "motivates hidden MMR and visible rank separation, not algorithm reconstruction",
                },
                {
                    "assumption": "Churn-risk proxy weights",
                    "status": "sensitivity-tested",
                    "grounding": "Weights are transparent configuration choices; proxy-weight variants test whether conclusions depend on one coefficient setting.",
                    "manuscript_use": "reported as synthetic proxy, not real churn",
                },
            ]
        ),
    )
    ablation_path = ROOT / "results/tables/ablation_study.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        base = ablation[ablation["scenario"] == "ablation_default"].set_index("seed")
        rows = []
        metrics = ["expected_winprob_deviation", "role_mismatch_rate", "mmr_rank_divergence", "churn_risk_proxy"]
        for scenario, group in ablation.groupby("scenario"):
            merged = group.set_index("seed").join(base[metrics], rsuffix="_base")
            row = {"scenario": scenario}
            for metric in metrics:
                row[metric] = float(group[metric].mean())
                if scenario != "ablation_default":
                    row[f"delta_{metric}"] = float((merged[metric] - merged[f"{metric}_base"]).mean())
                else:
                    row[f"delta_{metric}"] = 0.0
            rows.append(row)
        write_table("table_13_ablation_results.csv", pd.DataFrame(rows))
    robust_path = ROOT / "results/tables/population_robustness_results.csv"
    if robust_path.exists():
        robust = pd.read_csv(robust_path)
        write_table(
            "table_14_population_robustness.csv",
            robust.groupby("scenario", as_index=False)[
                ["expected_winprob_deviation", "role_mismatch_rate", "mmr_rank_divergence", "churn_risk_proxy"]
            ].mean(),
        )
    team = main[main["scenario"].eq("team_matchmaking")].copy()
    if not team.empty:
        rows = []
        metric = "expected_winprob_deviation"
        ref = team[team["policy"].eq("nearest")].set_index("seed")
        for policy, group in team.groupby("policy"):
            vals = group[metric].to_numpy()
            mean, low, high = bootstrap_ci(vals)
            row = {
                "comparison": f"{policy} vs nearest",
                "policy_mean": mean,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "cliffs_delta_vs_nearest": cliffs_delta(vals, ref[metric].to_numpy()),
            }
            if policy != "nearest":
                joined = group.set_index("seed").join(ref[[metric]], rsuffix="_nearest")
                diff = joined[metric] - joined[f"{metric}_nearest"]
                row["paired_mean_difference"] = float(diff.mean())
                row["cohens_dz"] = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else 0.0
            else:
                row["paired_mean_difference"] = 0.0
                row["cohens_dz"] = 0.0
            rows.append(row)
        write_table("table_15_statistical_effects.csv", pd.DataFrame(rows))
        rank_rows = []
        for seed, g in team.groupby("seed"):
            ranks = g.sort_values(metric).reset_index(drop=True)
            for idx, row in ranks.iterrows():
                rank_rows.append({"seed": seed, "policy": row["policy"], "rank": idx + 1})
        ranking = pd.DataFrame(rank_rows)
        write_table(
            "table_16_policy_ranking_stability.csv",
            ranking.groupby("policy", as_index=False).agg(mean_rank=("rank", "mean"), top_rank_count=("rank", lambda s: int((s == 1).sum()))),
        )
    write_table(
        "table_08_threats_to_validity.csv",
        pd.DataFrame(
            [
                {"threat": "Synthetic data only", "mitigation": "Explicitly framed as simulation evidence"},
                {"threat": "No proprietary logs", "mitigation": "No claims about commercial algorithm reconstruction"},
                {"threat": "Simplified churn proxy", "mitigation": "Reported as proxy, not real churn"},
                {"threat": "Simplified team synergy", "mitigation": "Sensitivity and limitations sections"},
            ]
        ),
    )
    riot_path = ROOT / "results/tables/riot_public_sources.csv"
    if riot_path.exists():
        riot = pd.read_csv(riot_path)
        write_table("table_09_riot_public_context.csv", riot)
    print("tables generated")
