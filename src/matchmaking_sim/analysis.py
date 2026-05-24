from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_frontier(df: pd.DataFrame, fairness_col="expected_winprob_deviation", wait_col="waiting_time_proxy") -> pd.DataFrame:
    points = df.sort_values([fairness_col, wait_col]).copy()
    efficient = []
    best_wait = float("inf")
    for _, row in points.iterrows():
        if row[wait_col] < best_wait:
            efficient.append(True)
            best_wait = row[wait_col]
        else:
            efficient.append(False)
    points["pareto_efficient"] = efficient
    return points


def cohens_d(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(1, len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
