from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matchmaking_sim.standard_reference_baselines import run_reference_rating_check

OUT = ROOT / "results" / "tables"


def _to_tex(df: pd.DataFrame, path: Path) -> None:
    display = df.copy()
    display["Baseline"] = display["baseline"].map(
        {
            "elo": "Elo",
            "dynamic_k": "Dynamic-K",
            "glicko_like_internal": "Internal Glicko-like",
            "standard_glicko2_reference": "Standard Glicko-2 reference",
        }
    )
    cols = ["Baseline", "rating_mae", "rating_rmse", "brier_score", "log_loss"]
    display = display[cols].rename(
        columns={
            "rating_mae": "MAE",
            "rating_rmse": "RMSE",
            "brier_score": "Brier",
            "log_loss": "Log loss",
        }
    )
    for col in ["MAE", "RMSE", "Brier", "Log loss"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [701, 702, 703, 704, 705]:
        rows.extend(run_reference_rating_check(seed=seed))
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "standard_reference_baseline_check_seed_level.csv", index=False)
    summary = (
        raw.groupby("baseline", as_index=False)
        .agg(
            rating_mae=("rating_mae", "mean"),
            rating_rmse=("rating_rmse", "mean"),
            brier_score=("brier_score", "mean"),
            log_loss=("log_loss", "mean"),
            n_seeds=("seed", "nunique"),
            n_players=("n_players", "first"),
            n_matches=("n_matches", "first"),
        )
        .sort_values("rating_mae")
    )
    summary.to_csv(OUT / "standard_reference_baseline_check.csv", index=False)
    _to_tex(summary, OUT / "standard_reference_baseline_check.tex")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
