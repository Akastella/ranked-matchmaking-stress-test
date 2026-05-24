from matchmaking_sim.skill_model import mmr_win_probability, win_probability


def test_probability_monotonicity():
    assert win_probability(1.0, 0.0) > 0.5
    assert mmr_win_probability(1600, 1500) > mmr_win_probability(1500, 1600)
