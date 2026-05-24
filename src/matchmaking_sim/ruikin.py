from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


TARGETS = {
    "perceived_unfairness_1_7": "感知不公平_1_7",
    "frustration_risk_1_7": "挫败风险_1_7",
    "progression_mismatch_salience_1_7": "段位错位显著性_1_7",
    "system_concern_1_7": "系统关注必要性_1_7",
    "acceptability_1_7": "是否可接受_1_7",
}

NUMERIC_FEATURES = [
    "fairness_debt_index",
    "unfair_match_exposure",
    "repeated_large_gap_exposure",
    "new_player_unfair_exposure",
    "low_activity_unfair_exposure",
    "smurf_victimization",
    "role_mismatch_rate",
    "tail_unfairness_p95",
    "rank_mmr_hysteresis_area",
    "mmr_rank_divergence",
    "waiting_time_proxy",
    "risk_transfer_indicator",
    "smurf_like_indicator",
    "new_player_indicator",
    "low_activity_indicator",
    "large_gap_indicator",
    "progression_mismatch_indicator",
]

CATEGORICAL_FEATURES = ["policy", "scenario", "stratum"]

MODEL_TARGETS = [
    "perceived_unfairness_1_7",
    "frustration_risk_1_7",
    "progression_mismatch_salience_1_7",
    "system_concern_1_7",
    "acceptability_reversed_1_7",
]


def load_ruikin_dataset(annotation_path: Path, session_features_path: Path) -> tuple[pd.DataFrame, dict]:
    ann = pd.read_csv(annotation_path)
    metrics = pd.read_csv(session_features_path)
    ann = ann[ann["data_source"].eq("human_annotation_real")].copy()
    metric_cols = metrics.drop(columns=["data_source"], errors="ignore").copy()
    metric_cols["has_direct_metric_match"] = 1
    merged = ann.merge(metric_cols, on="session_id", how="left", suffixes=("", "_metric"))

    def extract_number(text: str, pattern: str, default: float = 0.0) -> float:
        match = re.search(pattern, str(text))
        return float(match.group(1)) if match else default

    summary = merged["简短比赛序列摘要"].astype(str)
    reconstructed = pd.DataFrame(
        {
            "fairness_debt_index": [extract_number(t, r"公平债务指标约为\s*([0-9.]+)") for t in summary],
            "smurf_victimization": [extract_number(t, r"错配暴露约为\s*([0-9.]+)") for t in summary],
            "role_mismatch_rate": [extract_number(t, r"角色不匹配率约为\s*([0-9.]+)") for t in summary],
            "waiting_time_proxy": [extract_number(t, r"等待压力约为\s*([0-9.]+)") for t in summary],
            "rank_mmr_hysteresis_area": [extract_number(t, r"偏离面积约为\s*([0-9.]+)") for t in summary],
        }
    )
    reconstructed["mmr_rank_divergence"] = reconstructed["rank_mmr_hysteresis_area"] / 90.0
    reconstructed["unfair_match_exposure"] = reconstructed["fairness_debt_index"]
    reconstructed["repeated_large_gap_exposure"] = reconstructed["fairness_debt_index"]
    reconstructed["new_player_unfair_exposure"] = merged["该玩家是否新玩家"].astype(str).eq("是").astype(float) * reconstructed["fairness_debt_index"]
    reconstructed["low_activity_unfair_exposure"] = 0.0
    reconstructed["tail_unfairness_p95"] = reconstructed["fairness_debt_index"]

    for col in reconstructed.columns:
        if col not in merged:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(reconstructed[col])

    if "stratum" not in merged or merged["stratum"].isna().all():
        merged["stratum"] = "returned_annotation_reconstructed"
    else:
        high_fair = merged["fairness_debt_index"] >= merged["fairness_debt_index"].quantile(0.75)
        high_smurf = merged["smurf_victimization"] >= merged["smurf_victimization"].quantile(0.75)
        high_hyst = merged["rank_mmr_hysteresis_area"] >= merged["rank_mmr_hysteresis_area"].quantile(0.75)
        low_risk = (
            (merged["fairness_debt_index"] <= merged["fairness_debt_index"].quantile(0.25))
            & (merged["smurf_victimization"] <= merged["smurf_victimization"].quantile(0.25))
            & (merged["rank_mmr_hysteresis_area"] <= merged["rank_mmr_hysteresis_area"].quantile(0.25))
        )
        derived = np.select(
            [low_risk, high_smurf, high_hyst, high_fair],
            ["baseline_low_risk", "high_smurf_victimization", "high_rank_mmr_hysteresis", "high_fairness_debt"],
            default="ambiguous_case",
        )
        merged["stratum"] = merged["stratum"].fillna(pd.Series(derived, index=merged.index))

    for en, zh in TARGETS.items():
        merged[en] = pd.to_numeric(merged[zh], errors="coerce")
    merged["acceptability_reversed_1_7"] = 8.0 - merged["acceptability_1_7"]

    merged["risk_transfer_indicator"] = merged["stratum"].astype(str).eq("risk_redistribution_case").astype(float)
    merged["smurf_like_indicator"] = merged["是否存在疑似小号/隐藏实力错配"].astype(str).eq("是").astype(float)
    merged["new_player_indicator"] = merged["该玩家是否新玩家"].astype(str).eq("是").astype(float)
    merged["low_activity_indicator"] = merged["stratum"].astype(str).str.contains("low_activity", case=False, na=False).astype(float)
    merged["large_gap_indicator"] = merged["是否存在大实力差距"].astype(str).eq("是").astype(float)
    merged["progression_mismatch_indicator"] = merged["可见段位与隐藏估计是否明显错位"].astype(str).eq("是").astype(float)
    merged["annotator_id"] = merged["标注者编号"].astype(str)

    for col in NUMERIC_FEATURES:
        if col not in merged:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    for col in CATEGORICAL_FEATURES:
        merged[col] = merged[col].astype(str).fillna("unknown")

    report = {
        "human_rows": int(len(ann)),
        "metric_rows": int(len(metrics)),
        "matched_rows": int(len(merged)),
        "dropped_rows_without_metrics": 0,
        "rows_with_direct_session_metric_match": int(pd.to_numeric(merged.get("has_direct_metric_match", 0), errors="coerce").fillna(0).sum()),
        "rows_using_summary_reconstruction": int(len(merged) - pd.to_numeric(merged.get("has_direct_metric_match", 0), errors="coerce").fillna(0).sum()),
        "unique_annotators": int(merged["annotator_id"].nunique()),
        "unique_sessions": int(merged["session_id"].nunique()),
        "synthetic_annotator_rows_used": 0,
    }
    return merged, report


def make_preprocessor(include_categorical: bool = False) -> ColumnTransformer:
    transformers = [("num", StandardScaler(), NUMERIC_FEATURES)]
    if include_categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES))
    return ColumnTransformer(transformers)


def make_model(name: str, include_categorical: bool = False, monotonic: bool = False, seed: int = 20260517) -> Pipeline:
    pre = make_preprocessor(include_categorical=include_categorical)
    if name == "ruikin_l":
        model = Ridge(alpha=1.0)
    elif name == "ruikin_m":
        # Monotonic constraints are only valid for the numeric-only feature set.
        model = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.045,
            max_leaf_nodes=12,
            l2_regularization=0.02,
            monotonic_cst=[1] * len(NUMERIC_FEATURES) if monotonic else None,
            random_state=seed,
        )
    elif name == "ruikin_n":
        model = MLPRegressor(
            hidden_layer_sizes=(12,),
            alpha=0.02,
            early_stopping=True,
            validation_fraction=0.18,
            max_iter=800,
            random_state=seed,
        )
    elif name == "scenario_label_only":
        return Pipeline([("pre", make_preprocessor(include_categorical=True)), ("model", Ridge(alpha=1.0))])
    elif name == "dummy_mean":
        model = DummyRegressor(strategy="mean")
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("model", model)])


def ruikin0_score(df: pd.DataFrame, target: str) -> np.ndarray:
    def z(col: str) -> np.ndarray:
        arr = df[col].astype(float).to_numpy()
        span = arr.max() - arr.min()
        return np.zeros_like(arr) if span == 0 else (arr - arr.min()) / span

    exposure = np.mean(
        [
            z("fairness_debt_index"),
            z("unfair_match_exposure"),
            z("repeated_large_gap_exposure"),
            z("smurf_victimization"),
            z("tail_unfairness_p95"),
        ],
        axis=0,
    )
    progression = np.mean([z("rank_mmr_hysteresis_area"), z("mmr_rank_divergence"), z("progression_mismatch_indicator")], axis=0)
    queue = z("waiting_time_proxy")
    role = z("role_mismatch_rate")
    if target == "progression_mismatch_salience_1_7":
        score = 0.72 * progression + 0.18 * exposure + 0.10 * queue
    elif target == "frustration_risk_1_7":
        score = 0.45 * exposure + 0.25 * queue + 0.20 * progression + 0.10 * role
    elif target == "system_concern_1_7":
        score = 0.40 * exposure + 0.25 * z("smurf_victimization") + 0.20 * progression + 0.15 * role
    elif target == "acceptability_reversed_1_7":
        score = 0.48 * exposure + 0.22 * queue + 0.18 * progression + 0.12 * role
    else:
        score = 0.55 * exposure + 0.20 * z("smurf_victimization") + 0.15 * role + 0.10 * progression
    return 1.0 + 6.0 * np.clip(score, 0.0, 1.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rho = spearmanr(y_true, y_pred, nan_policy="omit").statistic
    tau = kendalltau(y_true, y_pred, nan_policy="omit").statistic
    return {
        "spearman": float(0.0 if np.isnan(rho) else rho),
        "kendall": float(0.0 if np.isnan(tau) else tau),
        "mae": float(mean_absolute_error(y_true, np.clip(y_pred, 1.0, 7.0))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cv_evaluate(df: pd.DataFrame, model_name: str, target: str, splitter, include_categorical: bool = False, groups=None) -> list[dict]:
    rows = []
    X = df[NUMERIC_FEATURES + (CATEGORICAL_FEATURES if include_categorical else [])]
    y = df[target].astype(float).to_numpy()
    if model_name == "ruikin_0":
        pred = ruikin0_score(df, target)
        metrics = evaluate_predictions(y, pred)
        metrics.update({"fold": "all", "model": model_name, "target": target, "n_test": len(df)})
        return [metrics]
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        model = make_model(model_name, include_categorical=include_categorical, monotonic=(model_name == "ruikin_m"))
        model.fit(train[X.columns], train[target].astype(float).to_numpy())
        pred = model.predict(test[X.columns])
        metrics = evaluate_predictions(test[target].astype(float).to_numpy(), pred)
        metrics.update({"fold": fold, "model": model_name, "target": target, "n_test": len(test)})
        rows.append(metrics)
    return rows


def baseline_score(df: pd.DataFrame, feature: str, target: str) -> np.ndarray:
    arr = df[feature].astype(float).to_numpy()
    if target == "acceptability_1_7":
        arr = -arr
    span = arr.max() - arr.min()
    scaled = np.zeros_like(arr) if span == 0 else (arr - arr.min()) / span
    return 1.0 + 6.0 * scaled


def permutation_importance_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES]
    y = df[target].astype(float).to_numpy()
    model = make_model("ruikin_l")
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=20, random_state=20260517, scoring="neg_mean_absolute_error")
    rows = []
    for feature, mean, std in zip(NUMERIC_FEATURES, result.importances_mean, result.importances_std):
        rows.append({"target": target, "feature": feature, "importance_mean": mean, "importance_std": std})
    return pd.DataFrame(rows).sort_values(["target", "importance_mean"], ascending=[True, False])


def monotonicity_check(df: pd.DataFrame, target: str, model_name: str = "ruikin_m") -> pd.DataFrame:
    X = df[NUMERIC_FEATURES]
    y = df[target].astype(float).to_numpy()
    model = make_model(model_name, monotonic=True)
    model.fit(X, y)
    base = X.median().to_frame().T
    rows = []
    for feature in NUMERIC_FEATURES:
        vals = np.linspace(df[feature].quantile(0.05), df[feature].quantile(0.95), 25)
        probe = pd.concat([base] * len(vals), ignore_index=True)
        probe[feature] = vals
        pred = model.predict(probe)
        violation_rate = float(np.mean(np.diff(pred) < -1e-6))
        rows.append({"target": target, "feature": feature, "monotonic_violation_rate": violation_rate})
    return pd.DataFrame(rows)
