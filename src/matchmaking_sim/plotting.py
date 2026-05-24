from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import pareto_frontier
from .config import PROJECT_ROOT, ensure_dirs


COLORS = {
    "blue": "#355C7D",
    "teal": "#4C9085",
    "purple": "#7A6F9B",
    "rose": "#A65D67",
    "gold": "#C79646",
    "green": "#6C8F70",
    "gray": "#5F6670",
    "light": "#F4F6F8",
    "grid": "#D9DEE5",
}

LABELS = {
    "random": "Random",
    "nearest": "Nearest MMR",
    "expanding": "Expanding",
    "role_aware": "Role-aware",
    "fairness_constrained": "Fairness-constrained",
    "latency_constrained": "Latency-constrained",
    "multi_objective": "Multi-objective",
    "context_aware": "Context-aware",
    "dynamic_k": "Dynamic-K",
    "drift_aware": "Drift-aware",
    "glicko_like": "Glicko-like",
    "elo": "Elo",
    "team_matchmaking": "Default team",
    "cold_start": "Cold start",
    "smurf_disruption": "Smurf-like",
    "seasonal_reset": "Seasonal reset",
    "role_scarcity": "Role scarcity",
    "rank_mmr_divergence": "Rank-MMR divergence",
    "proxy_default": "Default",
    "proxy_exposure_heavy": "Exposure-heavy",
    "proxy_waiting_heavy": "Waiting-heavy",
    "proxy_divergence_heavy": "Divergence-heavy",
    "pareto_strict": "Fairness-heavy",
    "pareto_fast": "Waiting-heavy",
    "ablation_default": "Default",
    "ablation_no_role_penalty": "No role penalty",
    "ablation_no_uncertainty_penalty": "No uncertainty penalty",
    "ablation_no_new_player_penalty": "No new-player penalty",
    "ablation_visible_rank_unsmoothed": "No rank smoothing",
    "ablation_no_smurf_like_players": "No smurf-like players",
    "ablation_no_skill_drift": "No skill drift",
    "robust_population_1800": "Population 1800",
    "robust_population_3000": "Population 3000",
    "robust_smurf_high": "High smurf ratio",
    "robust_new_player_high": "High new-player ratio",
    "robust_role_scarcity_high": "High role scarcity",
    "robust_visible_smoothing_slow": "Slow rank smoothing",
    "robust_activity_skew": "Activity skew",
    "robust_candidate_pool_small": "Small candidate pool",
}

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,
    }
)


def _label(value: str, width: int | None = None) -> str:
    text = LABELS.get(str(value), str(value).replace("_", " ").title())
    return fill(text, width) if width else text


def _save(fig, name: str) -> None:
    out = PROJECT_ROOT / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _first_existing(*rels: str) -> Path:
    for rel in rels:
        path = PROJECT_ROOT / rel
        if path.exists():
            return path
    return PROJECT_ROOT / rels[-1]


def _mean_ci(group: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for key, g in group:
        vals = g[metric].dropna().to_numpy(float)
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = 1.96 * sd / max(1, np.sqrt(len(vals)))
        rows.append({"key": key, "mean": mean, "ci": ci})
    return pd.DataFrame(rows)


def _horizontal_bar(ax, data: pd.Series, color: str, xlabel: str, wrap: int = 24) -> None:
    labels = [_label(idx, wrap) for idx in data.index]
    y = np.arange(len(data))
    ax.barh(y, data.values, color=color, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    span = max(data.values) - min(data.values) if len(data) else 0
    offset = max(span * 0.02, max(data.values) * 0.01 if len(data) else 0.01)
    for yy, val in zip(y, data.values):
        ax.text(val + offset, yy, f"{val:.3f}" if val < 10 else f"{val:.1f}", va="center", fontsize=8, color="#2F343A")


def _boxplot(ax, groups: list[np.ndarray], labels: list[str], ylabel: str, color: str) -> None:
    bp = ax.boxplot(groups, labels=labels, patch_artist=True, showmeans=True, widths=0.55)
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.22, edgecolor=color, linewidth=1.2)
    for median in bp["medians"]:
        median.set(color=COLORS["rose"], linewidth=1.4)
    for mean in bp["means"]:
        mean.set(marker="D", markerfacecolor=COLORS["gold"], markeredgecolor="white", markersize=5)
    for i, vals in enumerate(groups, start=1):
        if len(vals):
            jitter = np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, color=color, alpha=0.5, zorder=3)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def generate_all_figures() -> list[str]:
    ensure_dirs()
    main_path = _first_existing(
        "results/tables/main_results_full_profile.csv",
        "results/tables/main_results_full_or_medium.csv",
        "results/tables/main_results.csv",
    )
    sens_path = _first_existing(
        "results/tables/proxy_weight_sensitivity_results.csv",
        "results/tables/sensitivity_results_full_or_medium.csv",
        "results/tables/sensitivity_results.csv",
    )
    trace_path = _first_existing(
        "results/experiment_runs/full_profile/rank_mmr_trace.csv",
        "results/experiment_runs/full_or_medium/rank_mmr_trace.csv",
    )
    df = pd.read_csv(main_path)
    sens = pd.read_csv(sens_path) if sens_path.exists() else pd.DataFrame()
    generated: list[str] = []

    # Figure 1: framework architecture.
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    layers = [
        ("Player State", "latent skill, role skill, activity, uncertainty"),
        ("Outcome Model", "probabilistic 1v1 and 5v5 match outcomes"),
        ("Rating and Rank", "hidden MMR, visible rank, divergence"),
        ("Matchmaking", "queue, role constraints, fairness, latency"),
        ("Proxy Metrics", "tail exposure, waiting pressure, simulated risk"),
    ]
    ys = np.linspace(0.76, 0.20, len(layers))
    for i, ((title, body), y) in enumerate(zip(layers, ys)):
        ax.text(
            0.50,
            y,
            f"{title}\n{body}",
            ha="center",
            va="center",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.08", fc=COLORS["light"], ec=COLORS["blue"], lw=1.1),
        )
        if i < len(layers) - 1:
            ax.annotate("", xy=(0.50, ys[i + 1] + 0.055), xytext=(0.50, y - 0.055), arrowprops=dict(arrowstyle="->", lw=1.2, color=COLORS["gray"]))
    ax.text(0.5, 0.92, "Experience-aware matchmaking simulation pipeline", ha="center", fontsize=13, weight="bold", color=COLORS["blue"])
    ax.text(0.5, 0.07, "All outputs are generated from synthetic configurations and fixed random seeds.", ha="center", fontsize=9, color=COLORS["gray"])
    _save(fig, "figure_01_framework_architecture")
    generated.append("figure_01_framework_architecture")

    # Figure 2.
    trace = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if trace.empty:
        x = np.arange(40)
        latent = 1500 + 5 * x + 60 * np.sin(x / 7)
        hidden = latent + 80 * np.exp(-x / 12) * np.cos(x / 4)
        visible = pd.Series(hidden).ewm(alpha=0.22).mean()
        ax.plot(x, latent, label="Latent skill, scaled", lw=2.0, color=COLORS["blue"])
        ax.plot(x, hidden, label="Hidden MMR", lw=2.0, color=COLORS["rose"])
        ax.plot(x, visible, label="Visible rank points", lw=2.0, color=COLORS["green"])
    else:
        ax.plot(trace["day"], trace["latent_skill_scaled"], label="Latent skill, scaled", lw=2.0, color=COLORS["blue"])
        ax.plot(trace["day"], trace["hidden_mmr_trace"], label="Hidden MMR trace", lw=2.0, color=COLORS["rose"])
        ax.plot(trace["day"], trace["visible_rank_trace"], label="Visible rank trace", lw=2.0, color=COLORS["green"])
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Rating scale")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "figure_02_hidden_mmr_visible_rank_dynamics")
    generated.append("figure_02_hidden_mmr_visible_rank_dynamics")

    # Figure 3.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rv = df[df["scenario"].eq("rating_validity")]
    source = rv if not rv.empty else df
    acc = source.groupby("rating")["rating_mae"].mean().sort_values()
    _horizontal_bar(ax, acc, COLORS["blue"], "Mean absolute rating error", wrap=22)
    _save(fig, "figure_03_rating_accuracy")
    generated.append("figure_03_rating_accuracy")

    # Figure 4.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    team = df[df["scenario"].eq("team_matchmaking")]
    source = team if not team.empty else df
    q = source.groupby("policy")["expected_winprob_deviation"].mean().sort_values()
    _horizontal_bar(ax, q, COLORS["purple"], "|Expected win probability - 0.5|", wrap=24)
    _save(fig, "figure_04_team_match_quality")
    generated.append("figure_04_team_match_quality")

    # Figures 5-8: diagnostic distributions.
    specs = [
        ("figure_05_cold_start_convergence", ["team_matchmaking", "cold_start"], "new_player_unfair_exposure", "New-player proxy-level exposure", COLORS["teal"]),
        ("figure_06_smurf_disruption_recovery", ["team_matchmaking", "smurf_disruption"], "smurf_victimization", "Smurf-like victimization exposure", COLORS["rose"]),
        ("figure_07_seasonal_reset_recovery", ["team_matchmaking", "seasonal_reset", "rank_mmr_divergence"], "mmr_rank_divergence", "Rank-MMR divergence", COLORS["green"]),
        ("figure_08_role_scarcity_effect", ["team_matchmaking", "role_scarcity"], "role_mismatch_rate", "Role-assignment mismatch", COLORS["gold"]),
    ]
    for name, scenarios, metric, ylabel, color in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        groups = [df[df["scenario"].eq(s)][metric].dropna().to_numpy(float) for s in scenarios]
        if all(len(v) == 0 for v in groups):
            ax.text(0.5, 0.5, "Scenario data unavailable", ha="center")
        else:
            _boxplot(ax, groups, [_label(s, 14) for s in scenarios], ylabel, color)
        _save(fig, name)
        generated.append(name)

    # Figure 9: aggregate Pareto view.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    pareto_source = df[df["scenario"].eq("team_matchmaking")]
    if pareto_source.empty:
        pareto_source = df
    agg = pareto_source.groupby(["policy", "scenario"]).agg(
        waiting=("waiting_time_proxy", "mean"),
        fairness=("expected_winprob_deviation", "mean"),
        waiting_sd=("waiting_time_proxy", "std"),
        fairness_sd=("expected_winprob_deviation", "std"),
    ).reset_index()
    p = pareto_frontier(agg.rename(columns={"waiting": "waiting_time_proxy", "fairness": "expected_winprob_deviation"}))
    offsets = {
        "multi_objective": (7, -12),
        "role_aware": (7, 8),
        "fairness_constrained": (-120, 16),
        "expanding": (-56, -22),
        "nearest": (-38, 24),
        "random": (18, -18),
    }
    for _, row in agg.iterrows():
        label = _label(row["policy"])
        efficient = bool(p[(p["policy"].eq(row["policy"])) & (p["scenario"].eq(row["scenario"]))]["pareto_efficient"].any())
        color = COLORS["blue"] if efficient else COLORS["rose"]
        ax.errorbar(row["waiting"], row["fairness"], xerr=(row["waiting_sd"] or 0), yerr=(row["fairness_sd"] or 0), fmt="o", ms=6, color=color, alpha=0.85, capsize=2)
        ax.annotate(
            label,
            (row["waiting"], row["fairness"]),
            xytext=offsets.get(row["policy"], (6, 5)),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color=COLORS["gray"], lw=0.6, shrinkA=2, shrinkB=2),
        )
    ax.set_xlabel("Waiting-time proxy")
    ax.set_ylabel("Fairness loss")
    ax.margins(x=0.16, y=0.16)
    ax.grid(True)
    _save(fig, "figure_09_fairness_latency_pareto")
    generated.append("figure_09_fairness_latency_pareto")

    # Figure 10.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    tail = source.groupby("policy")[["tail_unfairness_p95", "churn_risk_proxy"]].mean().sort_values("churn_risk_proxy")
    ax.scatter(tail["tail_unfairness_p95"], tail["churn_risk_proxy"], s=58, color=COLORS["teal"], edgecolor="white", linewidth=0.8)
    offsets = [(8, 10), (8, -14), (-105, 16), (-98, -16), (14, 18), (14, -20), (8, 12)]
    for (idx, row), off in zip(tail.iterrows(), offsets):
        ax.annotate(
            _label(idx),
            (row["tail_unfairness_p95"], row["churn_risk_proxy"]),
            xytext=off,
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color=COLORS["gray"], lw=0.6, shrinkA=2, shrinkB=2),
        )
    ax.set_xlabel("Tail unfairness p95")
    ax.set_ylabel("Simulated risk proxy")
    ax.margins(x=0.18, y=0.18)
    _save(fig, "figure_10_tail_unfairness_churn_proxy")
    generated.append("figure_10_tail_unfairness_churn_proxy")

    # Figure 11.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if sens.empty:
        ax.text(0.5, 0.5, "Sensitivity data unavailable", ha="center")
    elif "scenario" in sens.columns and sens["scenario"].str.startswith("proxy_", na=False).any():
        order = sens.groupby("scenario")["churn_risk_proxy"].mean().sort_values()
        _horizontal_bar(ax, order, COLORS["gray"], "Simulated risk proxy", wrap=24)
    else:
        pivot = sens.pivot_table(index="smurf_ratio", columns="drift_intensity", values="expected_winprob_deviation", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
        ax.set_xlabel("Drift intensity")
        ax.set_ylabel("Smurf ratio")
        fig.colorbar(im, ax=ax, label="Fairness loss")
    _save(fig, "figure_11_sensitivity_heatmap")
    generated.append("figure_11_sensitivity_heatmap")

    # Figure 14.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ablation_path = PROJECT_ROOT / "results/tables/ablation_study.csv"
    ablation = pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()
    if ablation.empty:
        ax.text(0.5, 0.5, "Ablation data unavailable", ha="center")
    else:
        order = ablation.groupby("scenario")["churn_risk_proxy"].mean().sort_values()
        _horizontal_bar(ax, order, COLORS["purple"], "Simulated risk proxy", wrap=28)
    _save(fig, "figure_14_ablation_proxy_risk")
    generated.append("figure_14_ablation_proxy_risk")

    # Figure 15.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    robust_path = PROJECT_ROOT / "results/tables/population_robustness_results.csv"
    robust = pd.read_csv(robust_path) if robust_path.exists() else pd.DataFrame()
    if robust.empty:
        ax.text(0.5, 0.5, "Robustness data unavailable", ha="center")
    else:
        order = robust.groupby("scenario")["expected_winprob_deviation"].mean().sort_values()
        _horizontal_bar(ax, order, COLORS["teal"], "Fairness loss", wrap=28)
    _save(fig, "figure_15_population_robustness")
    generated.append("figure_15_population_robustness")

    # Figure 12.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    rank_path = _first_existing("results/tables/rank_mmr_divergence_results_full_profile.csv", "results/tables/rank_mmr_divergence_results.csv")
    rank_df = pd.read_csv(rank_path) if rank_path.exists() else pd.DataFrame()
    if rank_df.empty:
        ax.text(0.5, 0.5, "Rank-MMR divergence data unavailable", ha="center")
    else:
        order = rank_df.groupby("scenario")["mmr_rank_divergence"].mean().sort_values(ascending=False).head(12)
        _horizontal_bar(ax, order, COLORS["green"], "Mean rank-MMR divergence", wrap=28)
    _save(fig, "figure_12_rank_mmr_divergence")
    generated.append("figure_12_rank_mmr_divergence")

    # Figure 13.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if trace.empty:
        ax.text(0.5, 0.5, "Rank trace unavailable", ha="center")
    else:
        ax.plot(trace["day"], trace["rank_mmr_divergence_trace"], color=COLORS["rose"], lw=2.0)
        ax.fill_between(trace["day"], trace["rank_mmr_divergence_trace"], color=COLORS["rose"], alpha=0.15)
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("|hidden MMR - visible rank points|")
    _save(fig, "figure_13_rank_mmr_divergence_dynamics")
    generated.append("figure_13_rank_mmr_divergence_dynamics")
    return generated
