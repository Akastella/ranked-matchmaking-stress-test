from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matchmaking_sim.ruikin import NUMERIC_FEATURES  # noqa: E402


TABLES = ROOT / "results" / "tables"
VAL = ROOT / "validation"


def corr_stats(y: np.ndarray, x: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> dict:
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    rho = spearmanr(x, y, nan_policy="omit").statistic
    tau = kendalltau(x, y, nan_policy="omit").statistic
    boot_rho = []
    boot_tau = []
    if len(y) >= 10:
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            br = spearmanr(x[idx], y[idx], nan_policy="omit").statistic
            bt = kendalltau(x[idx], y[idx], nan_policy="omit").statistic
            if np.isfinite(br):
                boot_rho.append(br)
            if np.isfinite(bt):
                boot_tau.append(bt)
    return {
        "n_rows": int(len(y)),
        "spearman": float(0.0 if not np.isfinite(rho) else rho),
        "spearman_ci_low": float(np.percentile(boot_rho, 2.5)) if boot_rho else np.nan,
        "spearman_ci_high": float(np.percentile(boot_rho, 97.5)) if boot_rho else np.nan,
        "kendall": float(0.0 if not np.isfinite(tau) else tau),
        "kendall_ci_low": float(np.percentile(boot_tau, 2.5)) if boot_tau else np.nan,
        "kendall_ci_high": float(np.percentile(boot_tau, 97.5)) if boot_tau else np.nan,
    }


def oof_ruikin_l(df: pd.DataFrame, target: str) -> np.ndarray:
    x = df[NUMERIC_FEATURES].astype(float)
    y = df[target].astype(float).to_numpy()
    pred = np.zeros(len(df), dtype=float)
    splitter = KFold(n_splits=5, shuffle=True, random_state=20260517)
    for train_idx, test_idx in splitter.split(x):
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(x.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x.iloc[test_idx])
    return np.clip(pred, 1.0, 7.0)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(VAL / "ruikin_dataset.csv")
    direct = df[pd.to_numeric(df.get("has_direct_metric_match", 0), errors="coerce").fillna(0).gt(0)].copy()
    rng = np.random.default_rng(20260517)

    rows: list[dict] = []
    checks = [
        {
            "check": "fairness_debt_vs_perceived_unfairness",
            "predictor": "fairness_debt_index",
            "target": "perceived_unfairness_1_7",
            "predictor_source": "direct_session_metric",
        },
        {
            "check": "win_prob_deviation_proxy_vs_perceived_unfairness",
            "predictor": "tail_unfairness_p95",
            "target": "perceived_unfairness_1_7",
            "predictor_source": "direct_session_metric_proxy_for_balance_deviation",
        },
        {
            "check": "rank_mmr_hysteresis_vs_progression_mismatch",
            "predictor": "rank_mmr_hysteresis_area",
            "target": "progression_mismatch_salience_1_7",
            "predictor_source": "direct_session_metric",
        },
        {
            "check": "smurf_victimization_vs_system_concern",
            "predictor": "smurf_victimization",
            "target": "system_concern_1_7",
            "predictor_source": "direct_session_metric",
        },
    ]
    for check in checks:
        stats = corr_stats(
            direct[check["target"]].astype(float).to_numpy(),
            direct[check["predictor"]].astype(float).to_numpy(),
            rng,
        )
        stats.update(check)
        stats["model_type"] = "single_channel_proxy"
        rows.append(stats)

    for target in ["perceived_unfairness_1_7", "system_concern_1_7", "progression_mismatch_salience_1_7"]:
        pred = oof_ruikin_l(direct, target)
        stats = corr_stats(direct[target].astype(float).to_numpy(), pred, rng)
        stats.update(
            {
                "check": f"ruikin_l_oof_vs_{target}",
                "predictor": "ruikin_l_oof_prediction",
                "target": target,
                "predictor_source": "direct_rows_5fold_oof",
                "model_type": "ruikin_l_direct_oof",
            }
        )
        rows.append(stats)

    out = pd.DataFrame(rows)
    out["warning"] = np.where(out["n_rows"].lt(100), "small direct-matched sample; directional sanity check only", "")
    out.to_csv(TABLES / "ruikin_direct_matched_rows_check.csv", index=False)

    display_cols = ["check", "target", "spearman", "spearman_ci_low", "spearman_ci_high", "kendall", "kendall_ci_low", "kendall_ci_high", "n_rows", "warning"]
    table = out[display_cols].copy()
    table_md = ["| " + " | ".join(display_cols) + " |", "| " + " | ".join(["---"] * len(display_cols)) + " |"]
    for _, row in table.iterrows():
        vals = []
        for col in display_cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        table_md.append("| " + " | ".join(vals) + " |")

    md = [
        "# Ruikin Direct-Matched Rows Report",
        "",
        f"- Direct-matched rows: {len(direct)}",
        "- These rows have direct links to `validation/session_metric_features.csv` rather than relying on summary-feature reconstruction.",
        "- Results are interpreted as a directional sanity check only because n is small.",
        "- The win-probability row uses `tail_unfairness_p95` as the available direct balance-deviation proxy; session-level expected win-probability deviation is not present in the annotation feature table.",
        "",
        "\n".join(table_md),
    ]
    (VAL / "ruikin_direct_matched_rows_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote direct-matched Ruikin check for {len(direct)} rows.")


if __name__ == "__main__":
    main()
