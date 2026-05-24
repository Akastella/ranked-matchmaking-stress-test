from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIMPAT = ROOT / "manuscript_simpat"
FIG_DIR = SIMPAT / "figures"
SUPP_DIR = SIMPAT / "supplement"
TABLE_DIR = SIMPAT / "tables"
RESULTS_CSV_DIR = SIMPAT / "results_csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SUPP_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

POLICY_LABELS = {
    "nearest": "Nearest-MMR",
    "role_aware": "Role-aware",
    "multi_objective": "Multi-objective",
    "multi_objective_without_smurf_penalty": "Multi-objective\n(no smurf penalty)",
    "multi_objective_with_smurf_exposure_penalty": "Multi-objective\n+ smurf penalty",
    "fairness_constrained": "Fairness-constrained",
    "expanding": "Expanding window",
    "random": "Random",
}

RATING_LABELS = {
    "elo": "Elo",
    "dynamic_k": "Dynamic-K",
    "glicko_like": "Glicko-like",
    "drift_aware": "Drift-aware",
    "context_aware": "Context-aware",
}


UNCERTAINTY_TABLES = []


def savefig(fig, stem: str, directory: Path = FIG_DIR) -> None:
    fig.tight_layout()
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def summarize_interval(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
    figure: str,
    panel: str,
    source: str,
    interval_type: str,
) -> pd.DataFrame:
    """Return descriptive mean and 95% CI over existing seeds/task cells."""
    grouped = df.groupby(group_cols)[metric].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.rename(columns={"count": "n", "std": "sd"})
    grouped["se"] = grouped["sd"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped.loc[grouped["n"] <= 1, "se"] = np.nan
    grouped["ci95"] = 1.96 * grouped["se"]
    grouped["ci95"] = grouped["ci95"].fillna(0.0)
    grouped["ci_low"] = grouped["mean"] - grouped["ci95"]
    grouped["ci_high"] = grouped["mean"] + grouped["ci95"]
    grouped["metric"] = metric
    grouped["figure"] = figure
    grouped["panel"] = panel
    grouped["source"] = source
    grouped["interval_type"] = interval_type
    UNCERTAINTY_TABLES.append(grouped.copy())
    return grouped


def figure_3(main: pd.DataFrame) -> None:
    rating_rows = main[main["scenario"].eq("rating_validity")]
    if rating_rows.empty:
        rating_rows = main
    rating = summarize_interval(
        rating_rows,
        ["rating"],
        "rating_mae",
        "Figure 3",
        "Rating layer",
        "seed-level full-core summaries",
        "95% CI over seeds",
    ).sort_values("mean", ascending=True)
    rating["label"] = rating["rating"].map(RATING_LABELS).fillna(rating["rating"])

    exposure_rows = main[main["scenario"].eq("team_matchmaking")]
    if exposure_rows.empty:
        exposure_rows = main
    exposure = summarize_interval(
        exposure_rows,
        ["policy"],
        "fairness_debt_index",
        "Figure 3",
        "Team-formation layer",
        "seed-level full-core summaries",
        "95% CI over seeds",
    ).sort_values("mean", ascending=True)
    exposure["label"] = exposure["policy"].map(POLICY_LABELS).fillna(exposure["policy"])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7))
    y_rating = np.arange(len(rating))
    axes[0].barh(y_rating, rating["mean"], color="#496d89")
    axes[0].errorbar(
        rating["mean"],
        y_rating,
        xerr=rating["ci95"],
        fmt="none",
        ecolor="#1f2933",
        elinewidth=1.35,
        capsize=3,
    )
    axes[0].set_yticks(y_rating, rating["label"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Rating MAE (mean +/- 95% CI)")
    axes[0].set_title("Rating layer")
    axes[0].grid(axis="x", alpha=0.25)

    y_exposure = np.arange(len(exposure))
    axes[1].barh(y_exposure, exposure["mean"], color="#9a6f3a")
    axes[1].errorbar(
        exposure["mean"],
        y_exposure,
        xerr=exposure["ci95"],
        fmt="none",
        ecolor="#1f2933",
        elinewidth=1.35,
        capsize=3,
    )
    axes[1].set_yticks(y_exposure, exposure["label"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Exposure debt index (mean +/- 95% CI)")
    axes[1].set_title("Team-formation layer")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Rating accuracy and exposure diagnostics are separate outputs", y=1.03, fontsize=13)
    fig.text(
        0.02,
        -0.02,
        "Error bars show seed-level 95% CI under fixed synthetic assumptions.",
        fontsize=9,
        color="#445",
    )
    savefig(fig, "Figure_3")


def figure_4(risk: pd.DataFrame) -> None:
    metric = "smurf_victimization"
    order = [
        "nearest",
        "role_aware",
        "multi_objective_without_smurf_penalty",
        "multi_objective_with_smurf_exposure_penalty",
    ]
    grouped = summarize_interval(
        risk,
        ["smurf_ratio", "policy"],
        metric,
        "Figure 4",
        "Smurf-ratio stress curve",
        "risk-transfer task cells",
        "95% CI over task cells",
    ).sort_values(["policy", "smurf_ratio"])

    fig, ax = plt.subplots(figsize=(8.1, 4.8))
    colors = ["#4c6a7f", "#73946b", "#b07a3a", "#7a6fa6"]
    for color, policy in zip(colors, order):
        sub = grouped[grouped["policy"].eq(policy)]
        if sub.empty:
            continue
        ax.plot(
            sub["smurf_ratio"],
            sub["mean"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=POLICY_LABELS.get(policy, policy),
        )
        ax.fill_between(
            sub["smurf_ratio"].astype(float),
            sub["ci_low"].astype(float),
            sub["ci_high"].astype(float),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    ax.set_xlabel("Smurf-like mismatch ratio")
    ax.set_ylabel("Smurf-like victimization exposure")
    ax.set_title("Risk transfer under smurf-like mismatch stress")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.text(
        0.02,
        -0.02,
        "Bands show task-cell 95% CI under fixed synthetic assumptions.",
        fontsize=9,
        color="#445",
    )
    savefig(fig, "Figure_4")


def figure_5(risk: pd.DataFrame) -> None:
    order = [
        "nearest",
        "role_aware",
        "multi_objective_without_smurf_penalty",
        "multi_objective_with_smurf_exposure_penalty",
    ]
    fairness = summarize_interval(
        risk,
        ["policy"],
        "fairness_debt",
        "Figure 5",
        "Aggregate channel",
        "risk-transfer task cells",
        "95% CI over task cells",
    )
    smurf = summarize_interval(
        risk,
        ["policy"],
        "smurf_victimization",
        "Figure 5",
        "Subgroup channel",
        "risk-transfer task cells",
        "95% CI over task cells",
    )
    grouped = pd.merge(
        fairness[["policy", "mean", "ci95"]],
        smurf[["policy", "mean", "ci95"]],
        on="policy",
        suffixes=("_fairness", "_smurf"),
    )
    grouped = grouped[grouped["policy"].isin(order)].copy()
    grouped["policy_order"] = grouped["policy"].map({p: i for i, p in enumerate(order)})
    grouped = grouped.sort_values("policy_order")
    labels = grouped["policy"].map(POLICY_LABELS).fillna(grouped["policy"]).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.8), sharey=False)
    y = np.arange(len(grouped))

    axes[0].barh(y, grouped["mean_fairness"], color="#6f8f72")
    axes[0].errorbar(
        grouped["mean_fairness"],
        y,
        xerr=grouped["ci95_fairness"],
        fmt="none",
        ecolor="#1f2933",
        elinewidth=1.35,
        capsize=3,
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Aggregate exposure debt (mean +/- 95% CI)")
    axes[0].set_title("Aggregate channel")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y, grouped["mean_smurf"], color="#9a6f3a")
    axes[1].errorbar(
        grouped["mean_smurf"],
        y,
        xerr=grouped["ci95_smurf"],
        fmt="none",
        ecolor="#1f2933",
        elinewidth=1.35,
        capsize=3,
    )
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Smurf-like exposure (mean +/- 95% CI)")
    axes[1].set_title("Subgroup channel")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Objective variants can shift risk between channels", y=1.03, fontsize=13)
    fig.text(
        0.02,
        -0.02,
        "Error bars show task-cell 95% CI under fixed synthetic assumptions.",
        fontsize=9,
        color="#445",
    )
    savefig(fig, "Figure_5")


def figure_6_hysteresis() -> None:
    trace_paths = [
        ROOT / "results" / "experiment_runs" / "full_profile" / "rank_mmr_trace.csv",
        ROOT / "results" / "experiment_runs" / "full_or_medium" / "rank_mmr_trace.csv",
    ]
    trace_path = next((p for p in trace_paths if p.exists()), None)
    if trace_path is None:
        raise FileNotFoundError("No rank-MMR trace CSV found.")
    trace = pd.read_csv(trace_path)
    day = trace["day"]
    hidden = trace["hidden_mmr_trace"]
    visible = trace["visible_rank_trace"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(day, hidden, color="#1f6fb2", linewidth=2.1, label="Hidden MMR trace")
    ax.plot(day, visible, color="#bd5b6a", linewidth=2.1, label="Visible-rank trace")
    ax.fill_between(day, hidden, visible, color="#ead9b7", alpha=0.72, label="Hysteresis area")
    ax.set_title("Rank-MMR hysteresis trace")
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Scaled rating units")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.text(
        0.02,
        -0.02,
        "Representative synthetic trace; shaded area is an area-over-time lag diagnostic.",
        fontsize=9,
        color="#445",
    )
    savefig(fig, "Figure_6")
    for ext in ("pdf", "png"):
        src = FIG_DIR / f"Figure_6.{ext}"
        dst = FIG_DIR / f"Figure_6_rank_mmr_hysteresis.{ext}"
        dst.write_bytes(src.read_bytes())


def figure_7_ranking_stability(ranking: pd.DataFrame) -> None:
    metric_labels = {
        "fairness_debt": "Exposure debt",
        "role_mismatch_rate": "Role mismatch",
        "smurf_victimization": "Smurf-like exposure",
    }
    order = [
        "role_aware",
        "multi_objective",
        "multi_objective_without_smurf_penalty",
        "multi_objective_with_smurf_exposure_penalty",
        "nearest",
    ]
    sub = ranking[
        ranking["metric"].isin(metric_labels)
        & ranking["policy"].isin(order)
    ].copy()
    metrics = [m for m in metric_labels if m in sub["metric"].unique()]
    pivot = (
        sub.pivot_table(index="policy", columns="metric", values="mean_rank")
        .reindex(order)
    )
    pivot = pivot[metrics]

    labels = [POLICY_LABELS.get(p, p).replace("\n", " ") for p in pivot.index]
    col_labels = [metric_labels[m] for m in pivot.columns]
    data = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    vmax = np.nanmax(data)
    im = ax.imshow(data, cmap="cividis_r", aspect="auto", vmin=1.0, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Ranking stability is metric-specific")
    ax.tick_params(axis="x", labelrotation=0)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if np.isfinite(val):
                color = "white" if val > (1.0 + vmax) / 2 else "#152536"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Mean rank across compact cells (lower is better)")
    fig.text(
        0.01,
        -0.03,
        "Values summarize compact robustness cells under fixed synthetic assumptions; no external policy dominance is implied.",
        fontsize=9.5,
        color="#445",
    )
    savefig(fig, "Figure_7_ranking_stability")
    for ext in ("pdf", "png"):
        src = FIG_DIR / f"Figure_7_ranking_stability.{ext}"
        dst = FIG_DIR / f"Figure_7.{ext}"
        dst.write_bytes(src.read_bytes())


def figure_s1_weight_sensitivity(weight: pd.DataFrame) -> None:
    labels = {
        "equal_weight": "Equal",
        "smurf_heavy": "Smurf-heavy",
        "new_player_heavy": "New-player",
        "role_heavy": "Role-heavy",
        "tail_risk_heavy": "Tail-risk",
        "annotation_informed_exploratory": "Annotation-informed",
    }
    pivot = weight.pivot_table(index="policy", columns="weight_scheme", values="rank_lowest_is_best")
    pivot = pivot.reindex(index=["role_aware", "multi_objective", "nearest", "fairness_constrained", "expanding", "random"])
    pivot = pivot[[c for c in labels if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    im = ax.imshow(pivot.values, cmap="cividis_r", aspect="auto", vmin=1, vmax=np.nanmax(pivot.values))
    ax.set_xticks(np.arange(len(pivot.columns)), [labels.get(c, c) for c in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), [POLICY_LABELS.get(i, i) for i in pivot.index])
    ax.set_title("Exposure-debt ranking sensitivity (1 = lowest debt)")
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{int(val)}", ha="center", va="center", color="#111", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rank")
    fig.text(
        0.02,
        -0.02,
        "Lower rank is better. Full plot is placed in the supplement; main text reports a compact summary.",
        fontsize=9,
        color="#445",
    )
    savefig(fig, "Figure_S1_weight_sensitivity", SUPP_DIR)


def weight_summary_table(weight: pd.DataFrame, risk: pd.DataFrame) -> None:
    smurf_means = risk.groupby("policy")["smurf_victimization"].mean()
    worst_smurf = smurf_means.idxmax() if not smurf_means.empty else "not evaluated"
    rows = []
    for scheme, sub in weight.groupby("weight_scheme"):
        best = sub.sort_values("rank_lowest_is_best").iloc[0]
        rows.append(
            {
                "scheme": scheme.replace("_", " "),
                "best": POLICY_LABELS.get(best["policy"], best["policy"]).replace("\n", " "),
                "debt": float(best["weighted_debt"]),
                "worst": POLICY_LABELS.get(worst_smurf, worst_smurf).replace("\n", " "),
                "note": "Retained",
            }
        )
    rows = sorted(rows, key=lambda r: r["scheme"])

    tex_rows = "\n".join(
        f"{r['scheme'].title()} & {r['best']} & {r['debt']:.3f} & {r['worst']} & {r['note']} \\\\"
        for r in rows
    )
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Weight-sensitivity summary for exposure debt. Equal weighting is an inspectable default; the full ranking plot is provided in the supplement.}}
\label{{tab:weight-sensitivity-summary}}
\small
\begin{{tabularx}}{{\textwidth}}{{p{{0.21\textwidth}} p{{0.16\textwidth}} r p{{0.26\textwidth}} X}}
\toprule
Weight scheme & Lowest-debt policy & Debt & Highest smurf-like channel & Risk-transfer \\
\midrule
{tex_rows}
\bottomrule
\end{{tabularx}}
\end{{table}}
"""
    (TABLE_DIR / "weight_sensitivity_summary.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    main_df = pd.read_csv(ROOT / "results" / "tables" / "main_results_full_profile_reframed.csv")
    risk_df = pd.read_csv(ROOT / "results" / "tables" / "risk_transfer_experiment.csv")
    weight_df = pd.read_csv(ROOT / "results" / "tables" / "fairness_debt_weight_sensitivity.csv")
    ranking_df = pd.read_csv(ROOT / "results" / "tables" / "ranking_stability.csv")
    figure_3(main_df)
    figure_4(risk_df)
    figure_5(risk_df)
    figure_6_hysteresis()
    figure_7_ranking_stability(ranking_df)
    figure_s1_weight_sensitivity(weight_df)
    weight_summary_table(weight_df, risk_df)
    if UNCERTAINTY_TABLES:
        uncertainty = pd.concat(UNCERTAINTY_TABLES, ignore_index=True, sort=False)
        uncertainty.to_csv(RESULTS_CSV_DIR / "figure_uncertainty_summary.csv", index=False)


if __name__ == "__main__":
    main()
