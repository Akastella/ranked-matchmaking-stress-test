from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matchmaking_sim.ruikin import (  # noqa: E402
    CATEGORICAL_FEATURES,
    MODEL_TARGETS,
    NUMERIC_FEATURES,
    baseline_score,
    cv_evaluate,
    evaluate_predictions,
    load_ruikin_dataset,
    monotonicity_check,
    permutation_importance_table,
    ruikin0_score,
)


TABLES = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures" / "final"
VAL = ROOT / "validation"


def summarize_cv(rows: list[dict], out_name: str) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    raw.to_csv(TABLES / f"{out_name}_folds.csv", index=False)
    summary = (
        raw.groupby(["model", "target"], as_index=False)
        .agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            kendall_mean=("kendall", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            folds=("fold", "count"),
        )
        .sort_values(["target", "spearman_mean"], ascending=[True, False])
    )
    summary.to_csv(TABLES / f"{out_name}.csv", index=False)
    return summary


def run_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=20260517)
    for target in MODEL_TARGETS:
        for model in ["ruikin_0", "ruikin_l", "ruikin_m", "ruikin_n"]:
            rows.extend(cv_evaluate(df, model, target, kfold))
    cv = summarize_cv(rows, "ruikin_cv_results")

    group_rows = []
    groups = df["annotator_id"].astype(str).to_numpy()
    group_split = GroupKFold(n_splits=5)
    for target in MODEL_TARGETS:
        for model in ["ruikin_l", "ruikin_m", "ruikin_n"]:
            group_rows.extend(cv_evaluate(df, model, target, group_split, groups=groups))
    group = summarize_cv(group_rows, "ruikin_group_cv_results")
    return cv, group


def run_leave_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_rows = []
    for target in MODEL_TARGETS:
        splitter = LeaveOneGroupOut()
        scenario_rows.extend(cv_evaluate(df, "ruikin_l", target, splitter, groups=df["stratum"].astype(str).to_numpy()))
    scenario = pd.DataFrame(scenario_rows)
    scenario.to_csv(TABLES / "ruikin_leave_scenario_out.csv", index=False)

    policy_rows = []
    for target in MODEL_TARGETS:
        splitter = LeaveOneGroupOut()
        policy_rows.extend(cv_evaluate(df, "ruikin_l", target, splitter, groups=df["policy"].astype(str).to_numpy()))
    policy = pd.DataFrame(policy_rows)
    policy.to_csv(TABLES / "ruikin_leave_policy_out.csv", index=False)
    return scenario, policy


def run_baselines(df: pd.DataFrame) -> pd.DataFrame:
    baseline_features = {
        "win_prob_deviation_proxy": "tail_unfairness_p95",
        "equal_weight_fairness_debt": "fairness_debt_index",
        "rank_mmr_hysteresis": "rank_mmr_hysteresis_area",
        "smurf_victimization": "smurf_victimization",
        "exposure_risk_index_proxy": "fairness_debt_index",
    }
    rows = []
    for target in MODEL_TARGETS:
        y = df[target].astype(float).to_numpy()
        for label, feature in baseline_features.items():
            pred = baseline_score(df, feature, target)
            metrics = evaluate_predictions(y, pred)
            metrics.update({"model": label, "target": target, "feature": feature})
            rows.append(metrics)
        for model in ["ruikin_0", "ruikin_l", "ruikin_m", "ruikin_n"]:
            source = TABLES / "ruikin_cv_results.csv"
        pred0 = ruikin0_score(df, target)
        metrics = evaluate_predictions(y, pred0)
        metrics.update({"model": "ruikin_0", "target": target, "feature": "hand_specified_multi_feature"})
        rows.append(metrics)
    base = pd.DataFrame(rows)
    base.to_csv(TABLES / "ruikin_vs_baselines.csv", index=False)
    return base


def train_full_outputs(df: pd.DataFrame) -> None:
    coef_rows = []
    for target in MODEL_TARGETS:
        X = df[NUMERIC_FEATURES]
        y = df[target].astype(float).to_numpy()
        pre = StandardScaler()
        Xs = pre.fit_transform(X)
        model = Ridge(alpha=1.0).fit(Xs, y)
        for feature, coef in zip(NUMERIC_FEATURES, model.coef_):
            coef_rows.append({"target": target, "feature": feature, "standardized_coefficient": coef})
    pd.DataFrame(coef_rows).to_csv(TABLES / "ruikin_coefficients.csv", index=False)

    imp = pd.concat([permutation_importance_table(df, target) for target in MODEL_TARGETS], ignore_index=True)
    imp.to_csv(TABLES / "ruikin_feature_importance.csv", index=False)
    mono = pd.concat([monotonicity_check(df, target) for target in MODEL_TARGETS], ignore_index=True)
    mono.to_csv(TABLES / "ruikin_monotonicity_check.csv", index=False)

    # Full-data Ruikin outputs for downstream diagnostics.
    out = df[["session_id", "annotator_id", "policy", "scenario", "stratum"]].copy()
    out["R_F_fairness_diagnostic"] = ruikin0_score(df, "perceived_unfairness_1_7")
    out["R_A_affective_strain_proxy"] = ruikin0_score(df, "frustration_risk_1_7")
    out["R_P_progression_tension"] = ruikin0_score(df, "progression_mismatch_salience_1_7")
    out["R_S_stickiness_risk_proxy"] = ruikin0_score(df, "acceptability_reversed_1_7")
    subgroup_rank = (
        df["subgroup_exposure_inequality"].rank(pct=True)
        if "subgroup_exposure_inequality" in df.columns
        else df["smurf_victimization"].rank(pct=True)
    )
    out["R_T_risk_transfer_score"] = 1.0 + 6.0 * np.clip(
        0.55 * df["smurf_victimization"].rank(pct=True)
        + 0.25 * df["risk_transfer_indicator"]
        + 0.20 * subgroup_rank,
        0.0,
        1.0,
    )
    out["R_C_composite_diagnostic"] = out[
        [
            "R_F_fairness_diagnostic",
            "R_A_affective_strain_proxy",
            "R_P_progression_tension",
            "R_S_stickiness_risk_proxy",
            "R_T_risk_transfer_score",
        ]
    ].mean(axis=1)
    out.to_csv(TABLES / "ruikin_affective_stickiness_results.csv", index=False)


def negative_controls(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260517)
    rows = []
    for target in MODEL_TARGETS:
        y = df[target].astype(float).to_numpy()
        controls = {
            "shuffled_labels": rng.permutation(y),
            "random_features": rng.normal(size=len(y)),
            "shuffled_fairness_debt": rng.permutation(df["fairness_debt_index"].to_numpy()),
            "scenario_label_only": None,
            "annotator_id_only": None,
        }
        for name, pred_source in controls.items():
            if pred_source is not None:
                pred = 1.0 + 6.0 * ((pred_source - pred_source.min()) / (pred_source.max() - pred_source.min() + 1e-9))
            else:
                col = "scenario" if name == "scenario_label_only" else "annotator_id"
                pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [col])])
                model = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
                splitter = KFold(n_splits=5, shuffle=True, random_state=20260517)
                pred = np.zeros(len(y))
                for tr, te in splitter.split(df):
                    model.fit(df.iloc[tr][[col]], y[tr])
                    pred[te] = model.predict(df.iloc[te][[col]])
            metrics = evaluate_predictions(y, pred)
            metrics.update({"control": name, "target": target})
            rows.append(metrics)
    neg = pd.DataFrame(rows)
    neg.to_csv(TABLES / "ruikin_negative_controls.csv", index=False)
    return neg


def model_comparison(cv: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    cv2 = cv.copy()
    cv2["cv_type"] = "random_5fold"
    group2 = group.copy()
    group2["cv_type"] = "group_by_annotator"
    comp = pd.concat([cv2, group2], ignore_index=True)
    comp.to_csv(TABLES / "ruikin_model_comparison.csv", index=False)
    target_summary = (
        comp.sort_values(["target", "cv_type", "spearman_mean"], ascending=[True, True, False])
        .groupby(["target", "cv_type"], as_index=False)
        .first()
    )
    target_summary.to_csv(TABLES / "ruikin_targetwise_summary.csv", index=False)
    general = comp.groupby(["model", "cv_type"], as_index=False).agg(
        mean_spearman=("spearman_mean", "mean"),
        mean_mae=("mae_mean", "mean"),
        targets=("target", "nunique"),
    )
    general.to_csv(TABLES / "ruikin_generalization_summary.csv", index=False)

    baseline = pd.read_csv(TABLES / "ruikin_vs_baselines.csv")
    trained = comp[comp["cv_type"].eq("group_by_annotator")].copy()
    trained = trained.rename(columns={"spearman_mean": "spearman", "mae_mean": "mae"})
    trained["feature"] = "multi_feature_annotation_calibrated"
    trained = trained[["target", "model", "spearman", "mae", "feature"]]
    combined = pd.concat([baseline, trained], ignore_index=True)
    combined.to_csv(TABLES / "ruikin_vs_baselines.csv", index=False)
    return comp


def overfitting_report(cv: pd.DataFrame, group: pd.DataFrame, neg: pd.DataFrame) -> None:
    mlp = cv[cv["model"].eq("ruikin_n")]["spearman_mean"].mean()
    linear = cv[cv["model"].eq("ruikin_l")]["spearman_mean"].mean()
    group_linear = group[group["model"].eq("ruikin_l")]["spearman_mean"].mean()
    neg_mean = neg["spearman"].mean()
    text = f"""# Ruikin Overfitting Risk Report

- Random 5-fold mean Spearman, Ruikin-L: {linear:.3f}
- Random 5-fold mean Spearman, Ruikin-N exploratory MLP: {mlp:.3f}
- Group-aware mean Spearman, Ruikin-L: {group_linear:.3f}
- Negative-control mean Spearman: {neg_mean:.3f}

The tiny MLP is exploratory and is reported only as a supplementary comparator. It is not used as primary evidence. The interpretable linear and monotonic variants are the auditable calibration models for this pilot-scale diagnostic layer.
"""
    (VAL / "ruikin_overfitting_risk_report.md").write_text(text, encoding="utf-8")
    pd.DataFrame(
        [
            {"check": "ruikin_n_minus_ruikin_l_random_cv", "value": mlp - linear},
            {"check": "ruikin_l_group_cv_drop", "value": linear - group_linear},
            {"check": "negative_control_mean_spearman", "value": neg_mean},
        ]
    ).to_csv(TABLES / "ruikin_overfitting_check.csv", index=False)


def figures() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    compare = pd.read_csv(TABLES / "ruikin_vs_baselines.csv")
    labels = {
        "acceptability_reversed_1_7": "Acceptability\n(reversed)",
        "frustration_risk_1_7": "Strain\nproxy",
        "perceived_unfairness_1_7": "Perceived\nunfairness",
        "progression_mismatch_salience_1_7": "Progression\nmismatch",
        "system_concern_1_7": "System\nconcern",
    }
    model_labels = {
        "win_prob_deviation_proxy": "Win-prob\nproxy",
        "equal_weight_fairness_debt": "Fairness\ndebt",
        "rank_mmr_hysteresis": "Hysteresis",
        "ruikin_0": "Ruikin-0",
        "ruikin_l": "Ruikin-L",
        "ruikin_m": "Ruikin-M",
    }
    targets = ["perceived_unfairness_1_7", "system_concern_1_7", "progression_mismatch_salience_1_7"]
    models = list(model_labels)
    subset = compare[compare["target"].isin(targets) & compare["model"].isin(models)].copy()
    subset["model_label"] = subset["model"].map(model_labels)
    plot = (
        subset.sort_values("model")
        .pivot_table(index="target", columns="model_label", values="spearman", aggfunc="max")
        .rename(index=labels)
    )
    plot = plot[[model_labels[m] for m in models if model_labels[m] in plot.columns]]
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    plot.plot(kind="bar", ax=ax, color=["#B8C7D2", "#8EAA93", "#C6A26A", "#A8B6C0", "#356B8C", "#6D8F63"])
    ax.set_ylabel("Spearman with pilot annotation")
    ax.set_xlabel("Pilot annotation target")
    ax.set_title("Ruikin Diagnostics Compared with Baseline Proxies")
    ax.legend(title="Diagnostic", fontsize=7.5, ncol=3)
    ax.tick_params(axis="x", rotation=0, labelsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ruikin_vs_baselines.pdf")
    fig.savefig(FIG / "fig_ruikin_vs_baselines.png", dpi=220)
    plt.close(fig)

    imp = pd.read_csv(TABLES / "ruikin_feature_importance.csv")
    top = (
        imp.groupby("feature", as_index=False)["importance_mean"].mean()
        .sort_values("importance_mean", ascending=False)
        .head(10)
        .sort_values("importance_mean")
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(top["feature"], top["importance_mean"], color="#356B8C")
    ax.set_xlabel("Permutation importance (MAE reduction)")
    ax.set_title("Ruikin-L Feature Importance")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ruikin_feature_importance.pdf")
    fig.savefig(FIG / "fig_ruikin_feature_importance.png", dpi=220)
    plt.close(fig)

    neg = pd.read_csv(TABLES / "ruikin_negative_controls.csv")
    neg_plot = neg.groupby("control", as_index=False)["spearman"].mean().sort_values("spearman")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.barh(neg_plot["control"], neg_plot["spearman"], color="#8AA1B1")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Mean Spearman against target")
    ax.set_title("Ruikin Negative Controls")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ruikin_negative_controls.pdf")
    fig.savefig(FIG / "fig_ruikin_negative_controls.png", dpi=220)
    plt.close(fig)

    # Schematic as reproducible vector-style matplotlib figure.
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    ax.axis("off")
    boxes = [
        ("Exposure\nchannels", 0.13, 0.68, "#E8F0F5"),
        ("Progression\nsignals", 0.13, 0.47, "#E8F0F5"),
        ("Queue / system\nsignals", 0.13, 0.26, "#E8F0F5"),
        ("Ruikin\ncalibration layer", 0.48, 0.47, "#F3EFE4"),
        ("R_F Fairness\nR_A Strain\nR_P Progression\nR_S Stickiness-risk\nR_T Risk transfer", 0.78, 0.47, "#FFF0D8"),
    ]
    for text, x, y, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="#3B5B70", linewidth=1.3),
        )
    for x1, y1, x2, y2 in [(0.25, 0.68, 0.39, 0.52), (0.25, 0.47, 0.39, 0.47), (0.25, 0.26, 0.39, 0.42), (0.57, 0.47, 0.68, 0.47)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4B6678"))
    ax.text(0.56, 0.15, "Pilot human judgments of synthetic summaries calibrate diagnostics; no real emotion or retention prediction.", ha="center", fontsize=8.5, color="#4F5D66")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ruikin_schematic.pdf")
    fig.savefig(FIG / "fig_ruikin_schematic.png", dpi=220)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    VAL.mkdir(parents=True, exist_ok=True)
    df, report = load_ruikin_dataset(VAL / "human_annotation_cleaned.csv", VAL / "session_metric_features.csv")
    df.to_csv(VAL / "ruikin_dataset.csv", index=False)
    report_lines = ["# Ruikin Dataset Report", ""]
    for key, value in report.items():
        report_lines.append(f"- {key}: {value}")
    report_lines.append("")
    report_lines.append("Only `data_source=human_annotation_real` rows are used. Synthetic annotator rows are excluded.")
    (VAL / "ruikin_dataset_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    cv, group = run_cv(df)
    scenario, policy = run_leave_out(df)
    base = run_baselines(df)
    train_full_outputs(df)
    neg = negative_controls(df)
    comp = model_comparison(cv, group)
    overfitting_report(cv, group, neg)
    figures()

    # Small target-wise table for manuscript.
    best = comp[comp["cv_type"].eq("group_by_annotator")].sort_values(["target", "spearman_mean"], ascending=[True, False]).groupby("target", as_index=False).first()
    best[["target", "model", "spearman_mean", "kendall_mean", "mae_mean"]].to_latex(TABLES / "ruikin_targetwise_summary.tex", index=False, escape=False)
    print("Ruikin training/evaluation complete.")


if __name__ == "__main__":
    main()
