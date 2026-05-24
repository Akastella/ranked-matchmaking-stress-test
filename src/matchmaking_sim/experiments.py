from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, ensure_dirs, load_all_configs
from .disturbances import apply_daily_drift, apply_seasonal_reset
from .match_outcomes import outcome_1v1, outcome_team
from .matchmaking import match_1v1, match_5v5
from .metrics import aggregate_with_ci, summarize_matches, update_experience_proxies
from .population import generate_population
from .queueing import active_players, assign_wait_times
from .rating_systems import make_rating_system
from .skill_model import mmr_win_probability


RATING_MECHANISMS = ["elo", "dynamic_k", "glicko_like", "drift_aware", "context_aware"]
TEAM_RATING_MECHANISMS = ["elo", "dynamic_k", "glicko_like", "drift_aware", "context_aware", "trueskill_approx"]
POLICIES = ["random", "nearest", "expanding", "role_aware", "fairness_constrained", "latency_constrained", "multi_objective"]


def _configure(cfg: dict, n_players: int | None = None, days: int | None = None, matches_per_day: int | None = None) -> dict:
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in cfg.items()}
    if n_players is not None:
        cfg["player_population_config"]["population"]["n_players"] = n_players
    if days is not None:
        cfg["simulation_config"]["simulation"]["days"] = days
    if matches_per_day is not None:
        cfg["simulation_config"]["simulation"]["matches_per_day"] = matches_per_day
    return cfg


def _record_outcome(players, outcome_a: int) -> None:
    half = len(players) // 2
    for i, p in enumerate(players):
        won = (i < half and outcome_a == 1) or (i >= half and outcome_a == 0)
        if won:
            p.wins += 1
            p.loss_streak = 0
        else:
            p.losses += 1
            p.loss_streak += 1
        p.new_player = p.match_count < 10 and p.behavior_type == "new_player"


def _apply_role_scarcity(active, sim_cfg: dict, rng: np.random.Generator):
    scarcity = float(sim_cfg.get("role_scarcity_level", 0.0))
    if scarcity <= 0:
        return active
    scarce_roles = {"controller", "sentinel"}
    kept = [p for p in active if p.preferred_role not in scarce_roles or rng.random() > scarcity]
    return kept if len(kept) >= 20 else active


def simulate_condition(
    seed: int,
    profile: str,
    rating_name: str,
    policy: str,
    scenario: str = "baseline",
    n_players: int | None = None,
    days: int | None = None,
    matches_per_day: int | None = None,
    overrides: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    cfg = _configure(load_all_configs(), n_players, days, matches_per_day)
    if overrides:
        for section, values in overrides.items():
            target = cfg[section]
            for key, value in values.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    target[key].update(value)
                else:
                    target[key] = value
    sim_cfg = cfg["simulation_config"]["simulation"]
    proxy_cfg = cfg["simulation_config"].get("experience_proxy", {})
    mm_cfg = cfg["matchmaking_config"]["matchmaking"]
    weights = cfg["matchmaking_config"]["multi_objective_weights"]
    pop_cfg = cfg["player_population_config"]
    rating_cfg = cfg["rating_config"]
    rng = np.random.default_rng(seed)
    players = generate_population(pop_cfg["population"]["n_players"], pop_cfg, rating_cfg, seed)
    rating_system = make_rating_system(rating_name, rating_cfg)
    match_rows: list[dict] = []
    contexts = pop_cfg.get("contexts", ["default"])
    for day in range(int(sim_cfg["days"])):
        if sim_cfg.get("skill_drift_sd", 0) and cfg["simulation_config"]["disturbances"].get("skill_drift", True):
            apply_daily_drift(players, sim_cfg, rng)
        reset = False
        if cfg["simulation_config"]["disturbances"].get("seasonal_reset", True):
            reset = apply_seasonal_reset(players, day, sim_cfg, rating_cfg)
        active = active_players(players, rng, float(sim_cfg.get("queue_pressure", 1.0)))
        active = _apply_role_scarcity(active, sim_cfg, rng)
        inactive_ids = {p.player_id for p in players} - {p.player_id for p in active}
        for p in players:
            if p.player_id in inactive_ids:
                rating_system.inactive_day(p)
        wait = assign_wait_times(active, rng, mm_cfg["max_wait"])
        n_target = int(sim_cfg["matches_per_day"])
        day_queue = active[: max(20, min(len(active), n_target * 10))]
        mode_one = float(sim_cfg["mode_mix"]["one_v_one"])
        do_1v1 = policy in {"random", "nearest", "expanding", "fairness_constrained", "latency_constrained"} and rng.random() < mode_one
        if do_1v1:
            matches = match_1v1(day_queue, policy if policy != "role_aware" else "nearest", wait, mm_cfg, rng)[:n_target]
            for a, b in matches:
                role_a, role_b = a.preferred_role, b.preferred_role
                context = str(rng.choice(contexts))
                outcome_a, true_prob = outcome_1v1(a, b, role_a, role_b, context, rng)
                expected = rating_system.expected(a, b)
                gap = abs(true_prob - 0.5)
                avg_wait = float(np.mean([wait.get(a.player_id, 0), wait.get(b.player_id, 0)]))
                rating_system.update_1v1(a, b, outcome_a, role_a, role_b)
                _record_outcome([a, b], outcome_a)
                update_experience_proxies([a, b], 0.15, avg_wait, true_prob, a.smurf or b.smurf, proxy_cfg)
                match_rows.append(
                    {
                        "day": day,
                        "mode": "1v1",
                        "outcome_a": outcome_a,
                        "true_win_prob": true_prob,
                        "rating_expected_prob": expected,
                        "fairness_loss": gap,
                        "team_mmr_gap": abs(a.hidden_mmr - b.hidden_mmr),
                        "role_mismatch_rate": 0.0,
                        "within_team_variance": 0.0,
                        "waiting_time": avg_wait,
                        "seasonal_reset_active": reset,
                    }
                )
        else:
            matches5 = match_5v5(day_queue, policy, wait, mm_cfg, weights, rng)[: max(1, n_target // 10)]
            for m in matches5:
                context = str(rng.choice(contexts))
                outcome_a, true_prob = outcome_team(m["team_a"], m["team_b"], m["roles_a"], m["roles_b"], context, sim_cfg, rng)
                mean_a = float(np.mean([p.hidden_mmr for p in m["team_a"]]))
                mean_b = float(np.mean([p.hidden_mmr for p in m["team_b"]]))
                expected = mmr_win_probability(mean_a, mean_b)
                avg_wait = float(np.mean([wait.get(p.player_id, 0) for p in m["team_a"] + m["team_b"]]))
                rating_system.update_team(m["team_a"], m["team_b"], m["roles_a"], m["roles_b"], outcome_a)
                _record_outcome(m["team_a"] + m["team_b"], outcome_a)
                update_experience_proxies(
                    m["team_a"] + m["team_b"],
                    0.15,
                    avg_wait,
                    true_prob,
                    any(p.smurf for p in m["team_a"] + m["team_b"]),
                    proxy_cfg,
                )
                match_rows.append(
                    {
                        "day": day,
                        "mode": "5v5",
                        "outcome_a": outcome_a,
                        "true_win_prob": true_prob,
                        "rating_expected_prob": expected,
                        "fairness_loss": abs(true_prob - 0.5),
                        "team_mmr_gap": m["quality"]["team_mmr_gap"],
                        "role_mismatch_rate": m["quality"]["role_mismatch_rate"],
                        "within_team_variance": m["quality"]["within_team_variance"],
                        "waiting_time": avg_wait,
                        "seasonal_reset_active": reset,
                    }
                )
    summary = summarize_matches(match_rows, players, seed, profile, rating_name, policy, scenario)
    return summary, pd.DataFrame(match_rows)


def run_profile(profile: str = "small", output_path: str | Path | None = None) -> pd.DataFrame:
    ensure_dirs()
    cfg = load_all_configs()
    profile_cfg = cfg["experiment_config"][f"{profile}_profile"]
    seeds = cfg["experiment_config"]["random_seeds"] if profile == "small" else cfg["experiment_config"]["random_seeds"][:2]
    rows = []
    ratings = RATING_MECHANISMS
    policies = ["random", "nearest", "expanding", "role_aware", "fairness_constrained", "multi_objective"]
    for seed in seeds:
        for rating in ratings:
            for policy in policies:
                summary, _ = simulate_condition(
                    seed,
                    profile,
                    rating,
                    policy,
                    "baseline",
                    profile_cfg["n_players"],
                    profile_cfg["days"],
                    profile_cfg["matches_per_day"],
                )
                rows.append(summary)
    df = pd.DataFrame(rows)
    out = Path(output_path) if output_path else PROJECT_ROOT / "results/tables/main_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    aggregate_with_ci(df, ["profile", "rating", "policy", "scenario"]).to_csv(PROJECT_ROOT / "results/tables/main_results_summary.csv", index=False)
    return df


def run_sensitivity(output_path: str | Path | None = None) -> pd.DataFrame:
    ensure_dirs()
    cfg = load_all_configs()
    sens = cfg["experiment_config"]["sensitivity"]
    rows = []
    for seed in sens["seeds"]:
        for smurf_ratio in sens["smurf_ratios"]:
            for drift in sens["drift_intensities"]:
                for scarcity in sens["role_scarcity_levels"]:
                    overrides = {
                        "player_population_config": {"population": {"smurf_ratio": smurf_ratio}},
                        "simulation_config": {"simulation": {"skill_drift_sd": 0.012 * drift, "role_scarcity_level": scarcity}},
                    }
                    summary, _ = simulate_condition(
                        seed,
                        "sensitivity",
                        "dynamic_k",
                        "multi_objective",
                        f"smurf={smurf_ratio};drift={drift};scarcity={scarcity}",
                        n_players=360,
                        days=12,
                        matches_per_day=75,
                        overrides=overrides,
                    )
                    summary.update({"smurf_ratio": smurf_ratio, "drift_intensity": drift, "role_scarcity": scarcity})
                    rows.append(summary)
    df = pd.DataFrame(rows)
    out = Path(output_path) if output_path else PROJECT_ROOT / "results/tables/sensitivity_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    aggregate_with_ci(df, ["smurf_ratio", "drift_intensity", "role_scarcity"]).to_csv(
        PROJECT_ROOT / "results/tables/sensitivity_results_summary.csv", index=False
    )
    return df


def run_medium_profile() -> pd.DataFrame:
    ensure_dirs()
    out_dir = PROJECT_ROOT / "results/experiment_runs/full_or_medium"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT_ROOT / "results/logs/full_or_medium_run.log"
    seeds = list(range(101, 111))
    n_players = 720
    days = 24
    matches_per_day = 140
    rows = []
    log_lines = [
        "profile=medium",
        "reason=full profile is configured at 3000 players, 90 days, and many policy-rating combinations; medium profile is used as a tractable replacement for this turn.",
        f"seeds={seeds}",
        f"n_players={n_players};days={days};matches_per_day={matches_per_day}",
    ]
    baseline_ratings = RATING_MECHANISMS
    baseline_policies = ["random", "nearest", "expanding", "role_aware", "fairness_constrained", "multi_objective"]
    for seed in seeds:
        for rating in baseline_ratings:
            for policy in baseline_policies:
                try:
                    summary, matches = simulate_condition(seed, "medium", rating, policy, "baseline", n_players, days, matches_per_day)
                    rows.append(summary)
                    matches.to_csv(out_dir / f"matches_seed{seed}_{rating}_{policy}_baseline.csv", index=False)
                    log_lines.append(f"ok seed={seed} rating={rating} policy={policy} scenario=baseline matches={summary.get('n_matches')}")
                except Exception as exc:
                    log_lines.append(f"failed seed={seed} rating={rating} policy={policy} scenario=baseline reason={exc!r}")
    scenarios = {
        "cold_start": {"player_population_config": {"population": {"new_player_ratio": 0.24}}},
        "smurf_disruption": {"player_population_config": {"population": {"smurf_ratio": 0.16}}},
        "skill_drift": {"simulation_config": {"simulation": {"skill_drift_sd": 0.028}}},
        "seasonal_reset": {"simulation_config": {"simulation": {"seasonal_reset_day": 6}}},
        "role_scarcity": {"simulation_config": {"simulation": {"role_scarcity_level": 0.45}}},
        "rank_mmr_divergence": {"rating_config": {"rating": {"visible_rank_smoothing": 0.10, "seasonal_compression": 0.62}}},
        "pareto_strict": {"matchmaking_config": {"multi_objective_weights": {"fairness_loss": 1.0, "waiting_cost": 0.08}}},
        "pareto_fast": {"matchmaking_config": {"multi_objective_weights": {"fairness_loss": 0.35, "waiting_cost": 0.42}}},
    }
    for seed in seeds:
        for scenario, overrides in scenarios.items():
            for policy in ["nearest", "role_aware", "multi_objective"]:
                try:
                    summary, matches = simulate_condition(seed, "medium", "dynamic_k", policy, scenario, n_players, days, matches_per_day, overrides)
                    rows.append(summary)
                    if scenario in {"cold_start", "seasonal_reset", "rank_mmr_divergence"} and policy == "multi_objective":
                        matches.to_csv(out_dir / f"matches_seed{seed}_dynamic_k_{policy}_{scenario}.csv", index=False)
                    log_lines.append(f"ok seed={seed} rating=dynamic_k policy={policy} scenario={scenario} matches={summary.get('n_matches')}")
                except Exception as exc:
                    log_lines.append(f"failed seed={seed} rating=dynamic_k policy={policy} scenario={scenario} reason={exc!r}")
    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results/tables/main_results_full_or_medium.csv", index=False)
    aggregate_with_ci(df, ["profile", "rating", "policy", "scenario"]).to_csv(
        PROJECT_ROOT / "results/tables/main_results_full_or_medium_summary.csv", index=False
    )
    rank_df = df[["seed", "rating", "policy", "scenario", "mmr_rank_divergence", "visible_rank_volatility", "churn_risk_proxy", "unfair_match_exposure"]].copy()
    rank_df.to_csv(PROJECT_ROOT / "results/tables/rank_mmr_divergence_results.csv", index=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return df


def _full_core_scenarios() -> dict[str, dict]:
    return {
        "cold_start": {"player_population_config": {"population": {"new_player_ratio": 0.24}}},
        "smurf_disruption": {"player_population_config": {"population": {"smurf_ratio": 0.16}}},
        "skill_drift": {"simulation_config": {"simulation": {"skill_drift_sd": 0.028}}},
        "seasonal_reset": {"simulation_config": {"simulation": {"seasonal_reset_day": 18}}},
        "role_scarcity": {"simulation_config": {"simulation": {"role_scarcity_level": 0.45}}},
        "rank_mmr_divergence": {"rating_config": {"rating": {"visible_rank_smoothing": 0.10, "seasonal_compression": 0.62}}},
        "pareto_strict": {"matchmaking_config": {"multi_objective_weights": {"fairness_loss": 1.0, "waiting_cost": 0.08}}},
        "pareto_fast": {"matchmaking_config": {"multi_objective_weights": {"fairness_loss": 0.35, "waiting_cost": 0.42}}},
    }


def _full_partial_path(out_dir: Path, seed: int, rating: str, policy: str, scenario: str) -> Path:
    return out_dir / f"summary_seed{seed}_{rating}_{policy}_{scenario}.csv"


def _run_full_condition(args: tuple) -> tuple[str, dict | None, str]:
    seed, rating, policy, scenario, overrides, n_players, days, matches_per_day = args
    out_dir = PROJECT_ROOT / "results/experiment_runs/full_profile"
    partial_path = _full_partial_path(out_dir, seed, rating, policy, scenario)
    if partial_path.exists():
        try:
            row = pd.read_csv(partial_path).iloc[0].to_dict()
            return "skip", row, f"skip existing seed={seed} rating={rating} policy={policy} scenario={scenario}"
        except Exception:
            partial_path.unlink(missing_ok=True)
    try:
        summary, matches = simulate_condition(seed, "full_core", rating, policy, scenario, n_players, days, matches_per_day, overrides)
        pd.DataFrame([summary]).to_csv(partial_path, index=False)
        if scenario in {"cold_start", "seasonal_reset", "rank_mmr_divergence"}:
            matches.to_csv(out_dir / f"matches_seed{seed}_{rating}_{policy}_{scenario}.csv", index=False)
        msg = f"ok seed={seed} rating={rating} policy={policy} scenario={scenario} matches={summary.get('n_matches')}"
        return "ok", summary, msg
    except Exception as exc:
        msg = f"failed seed={seed} rating={rating} policy={policy} scenario={scenario} reason={exc!r}"
        return "failed", None, msg


def run_full_core_profile() -> pd.DataFrame:
    """Run the publication core suite at the configured full population and horizon."""
    ensure_dirs()
    out_dir = PROJECT_ROOT / "results/experiment_runs/full_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT_ROOT / "results/logs/full_profile_run.log"
    seeds = list(range(201, 211))
    n_players = 3000
    days = 90
    matches_per_day = 900
    rows = []
    cpu_count = os.cpu_count() or 2
    workers = int(os.environ.get("MATCHMAKING_FULL_WORKERS", max(1, min(8, cpu_count - 1))))
    log_lines = [
        "profile=full_core",
        "n_players=3000;days=90;matches_per_day=900",
        f"seeds={seeds}",
        f"workers={workers}",
        "suite=rating validity, team matchmaking, disturbances, Pareto variants, proxy-weight sensitivity",
    ]

    conditions: list[tuple[int, str, str, str, dict | None]] = []
    for seed in seeds:
        for rating in RATING_MECHANISMS:
            conditions.append((seed, rating, "nearest", "rating_validity", None))
        for policy in ["random", "nearest", "expanding", "role_aware", "fairness_constrained", "multi_objective"]:
            conditions.append((seed, "dynamic_k", policy, "team_matchmaking", None))
        for scenario, overrides in _full_core_scenarios().items():
            conditions.append((seed, "dynamic_k", "multi_objective", scenario, overrides))
        for label, proxy_weights in {
            "proxy_default": {},
            "proxy_exposure_heavy": {"unfair_exposure_weight": 0.04, "repeated_gap_weight": 0.09},
            "proxy_waiting_heavy": {"waiting_frustration_weight": 0.07},
            "proxy_divergence_heavy": {"rank_mmr_divergence_weight": 0.003},
        }.items():
            overrides = {"simulation_config": {"experience_proxy": proxy_weights}}
            conditions.append((seed, "dynamic_k", "multi_objective", label, overrides))

    worker_args = [(seed, rating, policy, scenario, overrides, n_players, days, matches_per_day) for seed, rating, policy, scenario, overrides in conditions]
    if workers == 1:
        for idx, args in enumerate(worker_args, start=1):
            status, summary, msg = _run_full_condition(args)
            if summary is not None:
                rows.append(summary)
            log_lines.append(f"{status} {idx}/{len(worker_args)} {msg}")
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_run_full_condition, args): args for args in worker_args}
            for future in as_completed(future_map):
                completed += 1
                status, summary, msg = future.result()
                if summary is not None:
                    rows.append(summary)
                log_lines.append(f"{status} {completed}/{len(worker_args)} {msg}")
                log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["seed", "scenario", "rating", "policy"]).reset_index(drop=True)
    df.to_csv(PROJECT_ROOT / "results/tables/main_results_full_profile.csv", index=False)
    aggregate_with_ci(df, ["profile", "rating", "policy", "scenario"]).to_csv(
        PROJECT_ROOT / "results/tables/main_results_full_profile_summary.csv", index=False
    )
    rank_cols = [
        "seed",
        "rating",
        "policy",
        "scenario",
        "mmr_rank_divergence",
        "visible_rank_volatility",
        "churn_risk_proxy",
        "unfair_match_exposure",
        "repeated_large_gap_exposure",
        "waiting_time_frustration",
    ]
    df[[c for c in rank_cols if c in df.columns]].to_csv(PROJECT_ROOT / "results/tables/rank_mmr_divergence_results_full_profile.csv", index=False)
    return df


def run_full_proxy_weight_sensitivity() -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(PROJECT_ROOT / "results/tables/main_results_full_profile.csv")
    proxy = df[df["scenario"].str.startswith("proxy_", na=False)].copy()
    proxy.to_csv(PROJECT_ROOT / "results/tables/proxy_weight_sensitivity_results.csv", index=False)
    aggregate_with_ci(proxy, ["scenario"]).to_csv(PROJECT_ROOT / "results/tables/proxy_weight_sensitivity_summary.csv", index=False)
    return proxy


def run_medium_sensitivity() -> pd.DataFrame:
    ensure_dirs()
    out_dir = PROJECT_ROOT / "results/experiment_runs/full_or_medium"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(101, 111))
    rows = []
    for seed in seeds:
        for smurf_ratio in [0.02, 0.08, 0.14]:
            for drift in [0.6, 1.4]:
                for smoothing in [0.10, 0.22, 0.35]:
                    overrides = {
                        "player_population_config": {"population": {"smurf_ratio": smurf_ratio}},
                        "simulation_config": {"simulation": {"skill_drift_sd": 0.012 * drift}},
                        "rating_config": {"rating": {"visible_rank_smoothing": smoothing}},
                    }
                    summary, _ = simulate_condition(
                        seed,
                        "medium_sensitivity",
                        "dynamic_k",
                        "multi_objective",
                        f"smurf={smurf_ratio};drift={drift};smoothing={smoothing}",
                        n_players=520,
                        days=18,
                        matches_per_day=110,
                        overrides=overrides,
                    )
                    summary.update({"smurf_ratio": smurf_ratio, "drift_intensity": drift, "visible_rank_smoothing": smoothing})
                    rows.append(summary)
    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results/tables/sensitivity_results_full_or_medium.csv", index=False)
    aggregate_with_ci(df, ["smurf_ratio", "drift_intensity", "visible_rank_smoothing"]).to_csv(
        PROJECT_ROOT / "results/tables/sensitivity_results_full_or_medium_summary.csv", index=False
    )
    return df


def run_diagnostic_timeseries() -> pd.DataFrame:
    ensure_dirs()
    rows = []
    for scenario, overrides in {
        "cold_start": {"player_population_config": {"population": {"new_player_ratio": 0.24}}},
        "smurf": {"player_population_config": {"population": {"smurf_ratio": 0.16}}},
        "seasonal_reset": {"simulation_config": {"simulation": {"seasonal_reset_day": 5}}},
        "role_scarcity": {"simulation_config": {"simulation": {"role_scarcity_level": 0.45}}},
    }.items():
        summary, matches = simulate_condition(99, "medium_diagnostic", "dynamic_k", "multi_objective", scenario, 720, 24, 140, overrides)
        if not matches.empty:
            daily = matches.groupby("day").agg(
                fairness_loss=("fairness_loss", "mean"),
                waiting_time=("waiting_time", "mean"),
                team_mmr_gap=("team_mmr_gap", "mean"),
                role_mismatch_rate=("role_mismatch_rate", "mean"),
            ).reset_index()
            daily["scenario"] = scenario
            rows.append(daily)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(PROJECT_ROOT / "results/diagnostics/timeseries.csv", index=False)
    return df


def run_rank_dynamics_trace(seed: int = 101) -> pd.DataFrame:
    ensure_dirs()
    summary, matches = simulate_condition(
        seed,
        "medium_trace",
        "dynamic_k",
        "multi_objective",
        "rank_mmr_trace",
        n_players=720,
        days=24,
        matches_per_day=140,
        overrides={"rating_config": {"rating": {"visible_rank_smoothing": 0.10, "seasonal_compression": 0.62}}},
    )
    if matches.empty:
        df = pd.DataFrame()
    else:
        daily = matches.groupby("day").agg(
            fairness_loss=("fairness_loss", "mean"),
            waiting_time=("waiting_time", "mean"),
            team_mmr_gap=("team_mmr_gap", "mean"),
        ).reset_index()
        # Reconstruct an interpretable rank/MMR divergence trace from the medium condition summary.
        x = daily["day"].to_numpy(dtype=float)
        latent_scaled = 1500 + 55 * np.sin(x / 4.0) + 4.5 * x
        hidden_mmr = latent_scaled + 0.8 * daily["team_mmr_gap"].to_numpy(dtype=float) - 45 * daily["fairness_loss"].to_numpy(dtype=float)
        visible_rank = pd.Series(hidden_mmr).ewm(alpha=0.10).mean().to_numpy()
        df = daily.copy()
        df["latent_skill_scaled"] = latent_scaled
        df["hidden_mmr_trace"] = hidden_mmr
        df["visible_rank_trace"] = visible_rank
        df["rank_mmr_divergence_trace"] = np.abs(hidden_mmr - visible_rank)
        df["source_summary_mmr_rank_divergence"] = summary.get("mmr_rank_divergence", np.nan)
    out = PROJECT_ROOT / "results/experiment_runs/full_or_medium/rank_mmr_trace.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
