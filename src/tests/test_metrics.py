from matchmaking_sim.metrics import aggregate_with_ci
import pandas as pd


def test_aggregate_with_ci_creates_columns():
    df = pd.DataFrame({"policy": ["a", "a", "b"], "seed": [1, 2, 1], "x": [1.0, 3.0, 2.0], "n_matches": [5, 5, 5]})
    out = aggregate_with_ci(df, ["policy"])
    assert "x_mean" in out.columns
    assert len(out) == 2
