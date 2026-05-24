from matchmaking_sim.rank_presentation import rank_name


def test_rank_name_bounds():
    cfg = {"floor": 800, "step": 100, "names": ["Iron", "Bronze", "Silver"]}
    assert rank_name(100, cfg) == "Iron"
    assert rank_name(5000, cfg) == "Silver"
