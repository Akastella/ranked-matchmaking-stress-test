from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def latent_to_mmr(skill: float) -> float:
    return 1500.0 + 400.0 * skill


DEFAULT_EXPERIENCE_PROXY = {
    "loss_streak_weight": 0.03,
    "unfair_exposure_weight": 0.025,
    "repeated_gap_weight": 0.06,
    "rank_mmr_divergence_weight": 0.0015,
    "waiting_frustration_weight": 0.035,
    "waiting_soft_threshold": 5.0,
}


def update_experience_proxies(
    players,
    unfair_gap: float,
    wait_time: float,
    expected_prob: float,
    involved_smurf: bool,
    proxy_cfg: dict | None = None,
) -> None:
    weights = DEFAULT_EXPERIENCE_PROXY | (proxy_cfg or {})
    unfair = abs(expected_prob - 0.5) > unfair_gap
    for p in players:
        if unfair:
            p.unfair_match_count += 1
            p.repeated_large_skill_gap_exposure += 1
        else:
            p.repeated_large_skill_gap_exposure = max(0, p.repeated_large_skill_gap_exposure - 1)
        if involved_smurf and not p.smurf:
            p.smurf_victimization += 1
        p.waiting_time_frustration += max(0.0, wait_time - float(weights["waiting_soft_threshold"])) / 10.0
        p.churn_risk_state = float(
            min(
                1.0,
                float(weights["loss_streak_weight"]) * p.loss_streak
                + float(weights["unfair_exposure_weight"]) * p.unfair_match_count
                + float(weights["repeated_gap_weight"]) * p.repeated_large_skill_gap_exposure
                + float(weights["rank_mmr_divergence_weight"]) * p.rank_mmr_divergence
                + float(weights["waiting_frustration_weight"]) * p.waiting_time_frustration,
            )
        )


def summarize_matches(match_rows: list[dict], players, seed: int, profile: str, rating: str, policy: str, scenario: str) -> dict:
    df = pd.DataFrame(match_rows)
    if df.empty:
        return {
            "profile": profile,
            "seed": seed,
            "rating": rating,
            "policy": policy,
            "scenario": scenario,
            "n_matches": 0,
        }
    y = df["outcome_a"].astype(int)
    probs = np.clip(df["rating_expected_prob"].astype(float), 1e-5, 1 - 1e-5)
    rating_errors = [abs(p.hidden_mmr - latent_to_mmr(p.latent_global_skill)) for p in players]
    volatilities = [p.volatility for p in players]
    churn = [p.churn_risk_state for p in players]
    repeated_gap = [p.repeated_large_skill_gap_exposure for p in players]
    waiting_frustration = [p.waiting_time_frustration for p in players]
    low_activity_players = [p for p in players if p.behavior_type == "low_activity_player"]
    unfair = df["fairness_loss"].to_numpy(dtype=float)
    tail_cut = max(1, int(0.05 * len(unfair)))
    worst = np.sort(unfair)[-tail_cut:]
    unfair_match_exposure = float(np.mean([p.unfair_match_count for p in players]))
    repeated_large_gap_exposure = float(np.mean(repeated_gap))
    smurf_victimization = float(np.mean([p.smurf_victimization for p in players]))
    waiting_time_frustration = float(np.mean(waiting_frustration))
    exposure_risk_index = float(np.mean(churn))
    return {
        "profile": profile,
        "seed": seed,
        "rating": rating,
        "policy": policy,
        "scenario": scenario,
        "n_matches": int(len(df)),
        "prediction_log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probs)),
        "rating_mae": float(np.mean(rating_errors)),
        "rating_rmse": float(np.sqrt(np.mean(np.square(rating_errors)))),
        "rating_volatility": float(np.mean(volatilities)),
        "expected_winprob_deviation": float(df["fairness_loss"].mean()),
        "team_mmr_gap": float(df["team_mmr_gap"].mean()),
        "role_mismatch_rate": float(df["role_mismatch_rate"].mean()),
        "within_team_variance": float(df["within_team_variance"].mean()),
        "waiting_time_proxy": float(df["waiting_time"].mean()),
        "unfair_match_exposure": unfair_match_exposure,
        "repeated_large_gap_exposure": repeated_large_gap_exposure,
        "new_player_unfair_exposure": float(np.mean([p.unfair_match_count for p in players if p.new_player] or [0])),
        "low_activity_unfair_exposure": float(np.mean([p.unfair_match_count for p in low_activity_players] or [0])),
        "smurf_victimization": smurf_victimization,
        "waiting_time_frustration": waiting_time_frustration,
        "tail_unfairness_p95": float(np.quantile(unfair, 0.95)),
        "cvar_unfairness_worst5": float(np.mean(worst)),
        "visible_rank_volatility": float(np.std([p.visible_rank_points for p in players])),
        "mmr_rank_divergence": float(np.mean([p.rank_mmr_divergence for p in players])),
        "exposure_risk_index": exposure_risk_index,
        "churn_risk_proxy": exposure_risk_index,
        "loss_streak_exposure": float(np.mean([p.loss_streak for p in players])),
    }


def aggregate_with_ci(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"seed", "n_matches"}
    ]
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        for m in metrics:
            vals = g[m].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}_median"] = float(np.median(vals))
            half = 1.96 * row[f"{m}_sd"] / max(1, np.sqrt(len(vals)))
            row[f"{m}_ci95_low"] = float(row[f"{m}_mean"] - half)
            row[f"{m}_ci95_high"] = float(row[f"{m}_mean"] + half)
        rows.append(row)
    return pd.DataFrame(rows)
