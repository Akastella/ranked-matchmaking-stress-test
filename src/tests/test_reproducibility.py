from matchmaking_sim.experiments import simulate_condition


def test_fixed_seed_reproducible_summary():
    a, _ = simulate_condition(101, "test", "elo", "nearest", n_players=80, days=2, matches_per_day=12)
    b, _ = simulate_condition(101, "test", "elo", "nearest", n_players=80, days=2, matches_per_day=12)
    assert a["rating_mae"] == b["rating_mae"]
    assert a["expected_winprob_deviation"] == b["expected_winprob_deviation"]
