from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures" / "final"


SCORE_MAP = {
    "perceived_unfairness": "感知不公平_1_7",
    "frustration_risk": "挫败风险_1_7",
    "progression_mismatch_salience": "段位错位显著性_1_7",
    "system_concern": "系统关注必要性_1_7",
    "acceptability": "是否可接受_1_7",
    "confidence": "判断置信度_1_7",
}


CORE_PAIRS = [
    ("fairness_debt_index", "perceived_unfairness", "Q1 fairness debt vs perceived unfairness"),
    ("fairness_debt_index", "frustration_risk", "Q1 fairness debt vs frustration risk"),
    ("fairness_debt_index", "system_concern", "Q1 fairness debt vs system concern"),
    ("fairness_debt_index", "acceptability", "Q1 fairness debt vs acceptability"),
    ("rank_mmr_hysteresis_area", "progression_mismatch_salience", "Q2 hysteresis vs progression mismatch"),
    ("rank_mmr_hysteresis_area", "system_concern", "Q2 hysteresis vs system concern"),
    ("smurf_victimization", "perceived_unfairness", "Q3 smurf exposure vs perceived unfairness"),
    ("smurf_victimization", "system_concern", "Q3 smurf exposure vs system concern"),
    ("waiting_time_proxy", "frustration_risk", "waiting pressure vs frustration risk"),
    ("role_mismatch_rate", "system_concern", "role mismatch vs system concern"),
    ("win_prob_deviation_proxy", "perceived_unfairness", "balance proxy vs perceived unfairness"),
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ann_path = VALIDATION / "human_annotation_cleaned.csv"
    feature_path = VALIDATION / "session_metric_features_from_returned_annotations.csv"
    fallback = VALIDATION / "session_metric_features.csv"
    if not ann_path.exists():
        raise SystemExit("No real human annotation file found. Skipping manuscript human-annotation analysis.")
    if feature_path.exists():
        features = pd.read_csv(feature_path)
    elif fallback.exists():
        features = pd.read_csv(fallback)
    else:
        raise FileNotFoundError("No session metric feature file found.")
    ann = pd.read_csv(ann_path)
    if "data_source" not in ann.columns or not (ann["data_source"] == "human_annotation_real").all():
        raise ValueError("human_annotation_cleaned.csv must have data_source=human_annotation_real for every row.")
    return ann, features


def canonicalize(ann: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "标注者编号": "annotator_id",
        "标注者类型": "annotator_type",
        "是否有竞技游戏经验": "competitive_game_experience",
        "是否玩过排位/天梯模式": "ranked_mode_experience",
        "主要问题来源": "primary_problem_source",
        "标注耗时_秒": "annotation_time_seconds_raw",
    }
    df = ann.rename(columns=rename).copy()
    for english, chinese in SCORE_MAP.items():
        df[english] = pd.to_numeric(df[chinese], errors="coerce")
    for flag in ["attention_check_pass", "careful_reading_pass", "quality_pass", "too_fast_flag", "low_confidence_flag"]:
        if flag in df.columns:
            df[flag] = df[flag].astype(str).str.lower().map({"true": True, "false": False}).fillna(df[flag].astype(bool))
    features = features.copy()
    if "win_prob_deviation" not in features.columns:
        # The returned annotation summaries do not contain immediate win-probability deviation.
        # This proxy is kept separate and reported as a limitation, not as the original balance metric.
        features["win_prob_deviation_proxy"] = features.get("tail_unfairness_p95", np.nan)
    merged = df.merge(features, on=["session_id", "scenario", "policy", "seed", "data_source"], how="left", suffixes=("", "_feature"))
    merged["derived_stratum"] = derive_strata(merged)
    return merged


def derive_strata(df: pd.DataFrame) -> pd.Series:
    q = {}
    for col in ["fairness_debt_index", "rank_mmr_hysteresis_area", "smurf_victimization", "role_mismatch_rate", "waiting_time_proxy"]:
        s = pd.to_numeric(df[col], errors="coerce")
        q[col] = {"lo": s.quantile(0.2), "midlo": s.quantile(0.4), "midhi": s.quantile(0.6), "hi": s.quantile(0.8)}
    labels: list[str] = []
    for _, r in df.iterrows():
        fd = r["fairness_debt_index"]
        hy = r["rank_mmr_hysteresis_area"]
        sm = r["smurf_victimization"]
        rm = r["role_mismatch_rate"]
        wt = r["waiting_time_proxy"]
        if fd <= q["fairness_debt_index"]["lo"] and hy <= q["rank_mmr_hysteresis_area"]["midlo"] and sm <= q["smurf_victimization"]["midlo"]:
            labels.append("baseline_low_risk")
        elif sm >= q["smurf_victimization"]["hi"] and fd <= q["fairness_debt_index"]["midhi"]:
            labels.append("risk_redistribution_case")
        elif sm >= q["smurf_victimization"]["hi"]:
            labels.append("high_smurf_victimization")
        elif fd >= q["fairness_debt_index"]["hi"]:
            labels.append("high_fairness_debt")
        elif hy >= q["rank_mmr_hysteresis_area"]["hi"]:
            labels.append("high_rank_mmr_hysteresis")
        elif rm >= q["role_mismatch_rate"]["hi"]:
            labels.append("high_role_mismatch")
        elif wt >= q["waiting_time_proxy"]["hi"]:
            labels.append("high_waiting_pressure")
        elif (
            q["fairness_debt_index"]["midlo"] <= fd <= q["fairness_debt_index"]["midhi"]
            and q["rank_mmr_hysteresis_area"]["midlo"] <= hy <= q["rank_mmr_hysteresis_area"]["midhi"]
        ):
            labels.append("ambiguous_case")
        else:
            labels.append("mixed_other")
    return pd.Series(labels, index=df.index)


def subset_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all_valid_rows": pd.Series(True, index=df.index),
        "attention_pass_rows": df.get("attention_check_pass", pd.Series(True, index=df.index)).astype(bool),
        "core_qc_rows": (
            df.get("attention_check_pass", pd.Series(True, index=df.index)).astype(bool)
            & df.get("careful_reading_pass", pd.Series(True, index=df.index)).astype(bool)
            & ~df.get("too_fast_flag", pd.Series(False, index=df.index)).astype(bool)
        ),
        "strict_qc_rows": df.get("quality_pass", pd.Series(True, index=df.index)).astype(bool),
    }


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn, n_boot: int = 1000, seed: int = 20260517) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        try:
            v = fn(x[sample], y[sample])
            if not np.isnan(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.quantile(vals, [0.025, 0.975]))


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, n_perm: int = 1000, seed: int = 20260517) -> float:
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        yy = rng.permutation(y)
        val = stats.spearmanr(x, yy, nan_policy="omit").statistic
        if abs(val) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def corr_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset_name, mask in subset_masks(df).items():
        sub = df.loc[mask].copy()
        for xcol, ycol, label in CORE_PAIRS:
            if xcol not in sub.columns or ycol not in sub.columns:
                continue
            xy = sub[[xcol, ycol]].dropna()
            if len(xy) < 5 or xy[xcol].nunique() < 2 or xy[ycol].nunique() < 2:
                continue
            x = xy[xcol].to_numpy(float)
            y = xy[ycol].to_numpy(float)
            spear = stats.spearmanr(x, y).statistic
            kend = stats.kendalltau(x, y).statistic
            ci = bootstrap_ci(x, y, lambda a, b: stats.spearmanr(a, b).statistic)
            p = permutation_p(x, y, spear, n_perm=500)
            rows.append(
                {
                    "subset": subset_name,
                    "question": label,
                    "feature": xcol,
                    "human_rating": ycol,
                    "n": len(xy),
                    "spearman": spear,
                    "spearman_ci_low": ci[0],
                    "spearman_ci_high": ci[1],
                    "kendall": kend,
                    "permutation_p": p,
                }
            )
    return pd.DataFrame(rows)


def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "rows", "value": len(df)},
        {"metric": "unique_annotators", "value": df["annotator_id"].nunique()},
        {"metric": "unique_sessions", "value": df["session_id"].nunique()},
        {"metric": "scenarios", "value": df["scenario"].nunique()},
        {"metric": "policies", "value": df["policy"].nunique()},
        {"metric": "mean_rows_per_annotator", "value": len(df) / max(df["annotator_id"].nunique(), 1)},
        {"metric": "sessions_with_repeated_ratings", "value": int((df.groupby("session_id").size() > 1).sum())},
    ]
    for col in SCORE_MAP:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.extend(
            [
                {"metric": f"{col}_mean", "value": s.mean()},
                {"metric": f"{col}_median", "value": s.median()},
                {"metric": f"{col}_std", "value": s.std()},
            ]
        )
    for name, mask in subset_masks(df).items():
        rows.append({"metric": f"{name}_n", "value": int(mask.sum())})
    return pd.DataFrame(rows)


def regression_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_specs = {
        "balance_proxy_only": ["win_prob_deviation_proxy"],
        "fairness_debt_only": ["fairness_debt_index"],
        "diagnostic_channels": [
            "win_prob_deviation_proxy",
            "fairness_debt_index",
            "smurf_victimization",
            "rank_mmr_hysteresis_area",
            "role_mismatch_rate",
            "waiting_time_proxy",
        ],
    }
    targets = ["perceived_unfairness", "frustration_risk", "progression_mismatch_salience", "system_concern"]
    for subset_name, mask in subset_masks(df).items():
        sub = df.loc[mask]
        for target in targets:
            for model, features in model_specs.items():
                cols = [target] + features
                d = sub[cols].replace([np.inf, -np.inf], np.nan).dropna()
                if len(d) < 10:
                    continue
                X = d[features].to_numpy(float)
                y = d[target].to_numpy(float)
                if np.all(np.nanstd(X, axis=0) == 0):
                    continue
                lr = LinearRegression()
                lr.fit(X, y)
                pred = lr.predict(X)
                r2 = r2_score(y, pred)
                adj = 1 - (1 - r2) * (len(y) - 1) / max(len(y) - X.shape[1] - 1, 1)
                rss = float(np.sum((y - pred) ** 2))
                aic = len(y) * math.log(max(rss / len(y), 1e-12)) + 2 * (X.shape[1] + 1)
                rows.append(
                    {
                        "subset": subset_name,
                        "target": target,
                        "model": model,
                        "n": len(y),
                        "r2": r2,
                        "adjusted_r2": adj,
                        "aic": aic,
                        "coefficients": "; ".join(f"{f}={c:.4g}" for f, c in zip(features, lr.coef_)),
                        "note": "Exploratory OLS sensitivity, not causal or ordinal validation.",
                    }
                )
    return pd.DataFrame(rows)


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def strata_tests(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["derived_stratum"] == "baseline_low_risk"]
    rows = []
    targets = ["perceived_unfairness", "system_concern", "progression_mismatch_salience", "confidence"]
    comparisons = [
        "high_fairness_debt",
        "high_smurf_victimization",
        "high_rank_mmr_hysteresis",
        "risk_redistribution_case",
        "ambiguous_case",
    ]
    for comp in comparisons:
        other = df[df["derived_stratum"] == comp]
        for target in targets:
            a = other[target].dropna()
            b = baseline[target].dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            diff = float(np.median(a) - np.median(b))
            rows.append(
                {
                    "comparison": f"{comp} vs baseline_low_risk",
                    "target": target,
                    "n_comparison": len(a),
                    "n_baseline": len(b),
                    "median_difference": diff,
                    "mann_whitney_p": u.pvalue,
                    "cliffs_delta": cliffs_delta(a, b),
                }
            )
    return pd.DataFrame(rows)


def problem_source_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["derived_stratum", "primary_problem_source"]).size().reset_index(name="n")
    counts["share"] = counts["n"] / counts.groupby("derived_stratum")["n"].transform("sum")
    return counts


def quality_sensitivity(corr: pd.DataFrame) -> pd.DataFrame:
    keep = corr[corr["question"].isin([
        "Q1 fairness debt vs perceived unfairness",
        "Q2 hysteresis vs progression mismatch",
        "Q3 smurf exposure vs perceived unfairness",
    ])]
    return keep[["subset", "question", "n", "spearman", "spearman_ci_low", "spearman_ci_high", "kendall", "permutation_p"]]


def write_identity_report(df: pd.DataFrame) -> None:
    sessions_per = df.groupby("session_id").size()
    lines = [
        "# Human Annotation Identity Report",
        "",
        f"- Data source values: {df['data_source'].value_counts().to_dict()}",
        f"- Total annotation rows: {len(df)}",
        f"- Unique annotators: {df['annotator_id'].nunique()}",
        f"- Unique sessions: {df['session_id'].nunique()}",
        f"- Unique scenarios: {df['scenario'].nunique()}",
        f"- Unique policies: {df['policy'].nunique()}",
        f"- Mean rows per annotator: {len(df) / max(df['annotator_id'].nunique(), 1):.2f}",
        f"- Rows after attention check: {int(df['attention_check_pass'].sum()) if 'attention_check_pass' in df else 'NA'}",
        f"- Rows after core QC: {int((df['attention_check_pass'].astype(bool) & df['careful_reading_pass'].astype(bool) & ~df['too_fast_flag'].astype(bool)).sum()) if {'attention_check_pass','careful_reading_pass','too_fast_flag'}.issubset(df.columns) else 'NA'}",
        f"- Rows after strict QC: {int(df['quality_pass'].sum()) if 'quality_pass' in df else 'NA'}",
        f"- Sessions with repeated ratings: {int((sessions_per > 1).sum())}",
        "",
        "Inter-rater reliability is not estimated because the returned file has one rating row per analyzed session and no repeated rating overlap.",
        "The rows are treated as pilot human annotation plausibility evidence, not full construct validation.",
    ]
    (VALIDATION / "human_annotation_identity_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_core_findings(corr: pd.DataFrame, reg: pd.DataFrame, strata: pd.DataFrame, problem: pd.DataFrame) -> None:
    raw = corr[corr["subset"] == "all_valid_rows"].set_index("question")
    def fmt(q: str) -> str:
        if q not in raw.index:
            return "not available"
        r = raw.loc[q]
        return f"Spearman {r['spearman']:.3f} [{r['spearman_ci_low']:.3f}, {r['spearman_ci_high']:.3f}], Kendall {r['kendall']:.3f}"
    lines = [
        "# Human Annotation Core Findings",
        "",
        "The returned 600-row pilot annotation file is analyzed as `human_annotation_real`. It is not merged with synthetic annotator outputs.",
        "",
        "## Core directional alignment",
        f"- Fairness debt vs perceived unfairness: {fmt('Q1 fairness debt vs perceived unfairness')}.",
        f"- Fairness debt vs frustration risk: {fmt('Q1 fairness debt vs frustration risk')}.",
        f"- Rank--MMR hysteresis vs progression mismatch salience: {fmt('Q2 hysteresis vs progression mismatch')}.",
        f"- Smurf-victimization exposure vs perceived unfairness: {fmt('Q3 smurf exposure vs perceived unfairness')}.",
        "",
        "## Balance-only comparison",
        "Immediate win-probability deviation is not directly present in the returned annotation summaries. The script therefore reports a `win_prob_deviation_proxy` based on the available tail-unfairness field and marks the regression as exploratory. In this returned feature reconstruction, the balance proxy is not independent of fairness debt, so incremental model comparisons should be interpreted as a limitation rather than strong evidence.",
        "",
        "## Inter-rater reliability",
        "No repeated session overlap is present, so Krippendorff alpha, ICC, and average pairwise agreement are not estimated.",
    ]
    (VALIDATION / "human_annotation_core_findings.md").write_text("\n".join(lines), encoding="utf-8")


def plot_alignment(df: pd.DataFrame, x: str, y: str, outstem: str, xlabel: str, ylabel: str) -> None:
    d = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(d[x], d[y], s=18, alpha=0.45, color="#376996", edgecolor="none")
    if len(d) > 2:
        z = np.polyfit(d[x], d[y], 1)
        xx = np.linspace(d[x].min(), d[x].max(), 100)
        ax.plot(xx, z[0] * xx + z[1], color="#B24C63", lw=2)
    rho = stats.spearmanr(d[x], d[y]).statistic if len(d) else np.nan
    ci = bootstrap_ci(d[x].to_numpy(float), d[y].to_numpy(float), lambda a, b: stats.spearmanr(a, b).statistic) if len(d) > 5 else (np.nan, np.nan)
    ax.text(0.02, 0.96, f"Spearman rho={rho:.3f}\n95% bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}]", transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Pilot human annotation over synthetic session summaries")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{outstem}.pdf")
    fig.savefig(FIGURES / f"{outstem}.png", dpi=220)
    plt.close(fig)


def plot_strata(df: pd.DataFrame) -> None:
    order = ["baseline_low_risk", "high_fairness_debt", "high_smurf_victimization", "high_rank_mmr_hysteresis", "risk_redistribution_case", "ambiguous_case"]
    d = df[df["derived_stratum"].isin(order)].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data = [d.loc[d["derived_stratum"] == s, "perceived_unfairness"].dropna().to_numpy() for s in order]
    ax.boxplot(data, tick_labels=[s.replace("_", "\n") for s in order], showfliers=False, patch_artist=True, boxprops=dict(facecolor="#D9E5EC", color="#4A6670"), medianprops=dict(color="#B24C63", lw=2))
    ax.set_ylabel("Perceived unfairness (1-7)")
    ax.set_title("Pilot annotation ratings by derived stress stratum")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_human_strata_results.pdf")
    fig.savefig(FIGURES / "fig_human_strata_results.png", dpi=220)
    plt.close(fig)


def plot_problem_source(problem: pd.DataFrame) -> None:
    selected = ["baseline_low_risk", "high_smurf_victimization", "high_rank_mmr_hysteresis", "risk_redistribution_case", "high_role_mismatch"]
    d = problem[problem["derived_stratum"].isin(selected)].copy()
    if d.empty:
        return
    label_map = {
        "实力差距": "Skill gap",
        "匹配实力差距": "Skill gap",
        "连续失败": "Loss streak",
        "疑似小号/隐藏实力错配": "Smurf-like mismatch",
        "疑似小号/隐藏分异常": "Smurf-like mismatch",
        "角色不匹配": "Role mismatch",
        "角色/位置不匹配": "Role mismatch",
        "等待时间": "Waiting time",
        "等待时间过长": "Waiting time",
        "段位展示与实际实力错位": "Rank display mismatch",
        "段位显示与实际表现不一致": "Rank display mismatch",
        "多个因素共同作用": "Multiple factors",
        "看不出明显问题": "No clear issue",
        "问题不明显": "No clear issue",
    }
    d["problem_source_en"] = d["primary_problem_source"].map(label_map).fillna(d["primary_problem_source"].astype(str).str.encode("ascii", "ignore").str.decode("ascii").replace("", "Other"))
    pivot = d.pivot_table(index="derived_stratum", columns="problem_source_en", values="share", fill_value=0)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bottom = np.zeros(len(pivot))
    palette = ["#376996", "#88A0A8", "#B24C63", "#D9A441", "#4B7F52", "#7A6C89", "#A0A0A0", "#C77D4A"]
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].to_numpy()
        ax.bar(np.arange(len(pivot)), vals, bottom=bottom, label=str(col), color=palette[i % len(palette)], width=0.72)
        bottom += vals
    ax.set_xticks(np.arange(len(pivot)))
    ax.set_xticklabels([s.replace("_", "\n") for s in pivot.index], fontsize=8)
    ax.set_ylabel("Share of responses")
    ax.set_title("Primary problem source by derived stress stratum")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_problem_source_distribution.pdf")
    fig.savefig(FIGURES / "fig_problem_source_distribution.png", dpi=220)
    plt.close(fig)


def write_latex_table(corr: pd.DataFrame) -> None:
    keep = corr[(corr["subset"] == "all_valid_rows") & corr["question"].isin([
        "Q1 fairness debt vs perceived unfairness",
        "Q1 fairness debt vs frustration risk",
        "Q1 fairness debt vs system concern",
        "Q2 hysteresis vs progression mismatch",
        "Q3 smurf exposure vs perceived unfairness",
        "Q3 smurf exposure vs system concern",
    ])].copy()
    keep["Spearman (95% CI)"] = keep.apply(lambda r: f"{r.spearman:.3f} [{r.spearman_ci_low:.3f}, {r.spearman_ci_high:.3f}]", axis=1)
    keep["Kendall"] = keep["kendall"].map(lambda x: f"{x:.3f}")
    keep = keep[["question", "n", "Spearman (95% CI)", "Kendall"]]
    keep = keep.rename(columns={"question": "Diagnostic pair", "n": "N"})
    tex = keep.to_latex(index=False, escape=True, column_format="p{0.46\\linewidth}rll")
    (TABLES / "human_annotation_core_alignment.tex").write_text(tex, encoding="utf-8")


def write_figures_tex() -> None:
    tex = r"""\begin{figure}[!htbp]
\centering
\includegraphics[width=0.72\linewidth]{../results/figures/final/fig_human_fairness_debt_alignment.pdf}
\caption{Pilot human annotation alignment between fairness debt and perceived unfairness over synthetic session summaries. The annotation data are human-entered pilot responses, but the figure is not a full construct-validation study because the design lacks repeated ratings per session.}
\label{fig:human-fairness-debt}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.72\linewidth]{../results/figures/final/fig_human_hysteresis_alignment.pdf}
\caption{Pilot human annotation alignment between rank--MMR hysteresis and perceived progression-mismatch salience. The trace supports directional interpretability of the diagnostic, not validation against real gameplay telemetry.}
\label{fig:human-hysteresis}
\end{figure}
"""
    (ROOT / "manuscript" / "human_annotation_figures_final.tex").write_text(tex, encoding="utf-8")


def write_revised_report(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    raw = corr[corr["subset"] == "all_valid_rows"].set_index("question")
    lines = [
        "# Revised Human Annotation Validity Report",
        "",
        f"Rows imported: {len(df)}",
        f"Unique annotators: {df['annotator_id'].nunique()}",
        f"Unique sessions: {df['session_id'].nunique()}",
        f"Scenario coverage: {df['scenario'].nunique()} scenarios",
        f"Policy coverage: {df['policy'].nunique()} policies",
        f"Average rows per annotator: {len(df) / max(df['annotator_id'].nunique(), 1):.2f}",
        f"QC subset sizes: { {name: int(mask.sum()) for name, mask in subset_masks(df).items()} }",
        f"Sessions with repeated ratings: {int((df.groupby('session_id').size() > 1).sum())}",
        "",
        "Inter-rater reliability is not estimable because the returned file has no repeated session overlap.",
        "",
        "Core correlations, all valid rows:",
    ]
    for q in [
        "Q1 fairness debt vs perceived unfairness",
        "Q1 fairness debt vs frustration risk",
        "Q2 hysteresis vs progression mismatch",
        "Q3 smurf exposure vs perceived unfairness",
        "balance proxy vs perceived unfairness",
    ]:
        if q in raw.index:
            r = raw.loc[q]
            lines.append(f"- {q}: Spearman {r['spearman']:.3f} [{r['spearman_ci_low']:.3f}, {r['spearman_ci_high']:.3f}], Kendall {r['kendall']:.3f}.")
    lines.extend([
        "",
        "The balance-only comparison is limited because the returned summary features do not contain an independent immediate win-probability-deviation field. The available proxy is reconstructed from tail-unfairness summaries and is not a substitute for the original match-balance metric.",
        "",
        "Interpretation status: preliminary human annotation plausibility evidence. Do not claim full construct validation.",
    ])
    (VALIDATION / "human_annotation_validity_report_revised.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ann, features = load_inputs()
    df = canonicalize(ann, features)
    df.to_csv(VALIDATION / "human_annotation_analysis_dataset.csv", index=False, encoding="utf-8-sig")

    write_identity_report(df)
    desc = descriptives(df)
    corr = corr_rows(df)
    reg = regression_models(df)
    strata = strata_tests(df)
    problem = problem_source_distribution(df)
    quality = quality_sensitivity(corr)

    desc.to_csv(TABLES / "human_annotation_descriptives.csv", index=False)
    corr.to_csv(TABLES / "human_annotation_core_alignment.csv", index=False)
    reg.to_csv(TABLES / "human_annotation_regression_models.csv", index=False)
    strata.to_csv(TABLES / "human_annotation_strata_tests.csv", index=False)
    quality.to_csv(TABLES / "human_annotation_quality_sensitivity.csv", index=False)
    problem.to_csv(TABLES / "human_annotation_problem_source_distribution.csv", index=False, encoding="utf-8-sig")
    write_latex_table(corr)
    write_core_findings(corr, reg, strata, problem)
    write_revised_report(df, corr)

    plot_alignment(df, "fairness_debt_index", "perceived_unfairness", "fig_human_fairness_debt_alignment", "Fairness debt index", "Perceived unfairness (1-7)")
    plot_alignment(df, "rank_mmr_hysteresis_area", "progression_mismatch_salience", "fig_human_hysteresis_alignment", "Rank--MMR hysteresis area", "Progression mismatch salience (1-7)")
    plot_alignment(df, "smurf_victimization", "system_concern", "fig_human_smurf_concern_alignment", "Smurf-like exposure", "System concern (1-7)")
    plot_strata(df)
    plot_problem_source(problem)
    write_figures_tex()

    print("Human annotation manuscript analysis complete.")
    print(TABLES / "human_annotation_core_alignment.csv")


if __name__ == "__main__":
    main()
