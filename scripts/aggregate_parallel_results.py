from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from matchmaking_sim.metrics import aggregate_with_ci
from matchmaking_sim.standard_baselines import sanity_check_series


def _safe_read(name: str) -> pd.DataFrame:
    path = ROOT / "results/tables" / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fairness_debt(df: pd.DataFrame) -> pd.Series:
    cols = [
        "unfair_match_exposure",
        "repeated_large_gap_exposure",
        "new_player_unfair_exposure",
        "low_activity_unfair_exposure",
        "smurf_victimization",
        "role_mismatch_rate",
        "tail_unfairness_p95",
    ]
    vals = []
    for c in cols:
        x = df[c].astype(float)
        denom = x.max() - x.min()
        vals.append((x - x.min()) / denom if denom > 1e-12 else x * 0)
    return sum(vals) / len(vals)


def aggregate_risk_transfer() -> None:
    df = _safe_read("risk_transfer_experiment.csv")
    if df.empty:
        return
    df = df.drop(columns=[c for c in ["nearest_debt", "nearest_smurf"] if c in df.columns])
    df["fairness_debt"] = _fairness_debt(df)
    base = df[df["policy"].eq("nearest")].groupby(["seed", "smurf_ratio", "smurf_skill_gap", "role_scarcity"]).agg(
        nearest_debt=("fairness_debt", "mean"),
        nearest_smurf=("smurf_victimization", "mean"),
    )
    joined = df.join(base, on=["seed", "smurf_ratio", "smurf_skill_gap", "role_scarcity"])
    joined["fairness_debt_reduction_vs_nearest"] = joined["nearest_debt"] - joined["fairness_debt"]
    joined["smurf_victimization_delta_vs_nearest"] = joined["smurf_victimization"] - joined["nearest_smurf"]
    joined["risk_transfer_index"] = joined["smurf_victimization_delta_vs_nearest"] / joined["fairness_debt_reduction_vs_nearest"].abs().clip(lower=1e-6)
    joined.to_csv(ROOT / "results/tables/risk_transfer_experiment.csv", index=False)
    summary = aggregate_with_ci(joined, ["policy", "smurf_ratio", "smurf_skill_gap", "role_scarcity"])
    summary.to_csv(ROOT / "results/tables/risk_transfer_summary.csv", index=False)


def aggregate_robustness() -> None:
    for fname, group_cols, out in [
        ("large_robustness_results.csv", ["policy", "scenario"], "large_robustness_summary.csv"),
        ("stress_grid_results.csv", ["weight_set", "smurf_ratio", "role_scarcity", "visible_rank_smoothing", "candidate_pool_size"], "parameter_sensitivity_summary.csv"),
    ]:
        df = _safe_read(fname)
        if not df.empty:
            df["fairness_debt"] = _fairness_debt(df)
            df.to_csv(ROOT / "results/tables" / fname, index=False)
            aggregate_with_ci(df, group_cols).to_csv(ROOT / "results/tables" / out, index=False)


def ranking_stability() -> None:
    frames = []
    for fname in ["large_robustness_results.csv", "risk_transfer_experiment.csv"]:
        df = _safe_read(fname)
        if not df.empty:
            if "fairness_debt" not in df:
                df["fairness_debt"] = _fairness_debt(df)
            frames.append(df)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for metric, ascending in [("fairness_debt", True), ("smurf_victimization", True), ("role_mismatch_rate", True)]:
        for key, g in df.groupby(["seed", "scenario"], dropna=False):
            rank = g.groupby("policy")[metric].mean().rank(method="min", ascending=ascending)
            worst_rank = float(rank.max())
            for policy, value in rank.items():
                rows.append({"metric": metric, "seed": key[0], "scenario": key[1], "policy": policy, "rank": value, "is_worst": value == worst_rank})
    ranks = pd.DataFrame(rows)
    summary = ranks.groupby(["metric", "policy"]).agg(
        mean_rank=("rank", "mean"),
        first_place_rate=("rank", lambda x: float(np.mean(np.asarray(x) == 1))),
        worst_place_rate=("is_worst", "mean"),
        n=("rank", "size"),
    ).reset_index()
    summary.to_csv(ROOT / "results/tables/ranking_stability.csv", index=False)


def standard_baseline() -> None:
    pd.DataFrame(sanity_check_series()).to_csv(ROOT / "results/tables/standard_baseline_sanity_check.csv", index=False)


if __name__ == "__main__":
    aggregate_risk_transfer()
    aggregate_robustness()
    ranking_stability()
    standard_baseline()
    print("aggregated parallel results")
