from matchmaking_sim.config import load_all_configs
from matchmaking_sim.matchmaking import match_1v1, match_5v5
from matchmaking_sim.population import generate_population


def test_matchmakers_return_valid_groups():
    cfg = load_all_configs()
    players = generate_population(40, cfg["player_population_config"], cfg["rating_config"], 5)
    wait = {p.player_id: 0 for p in players}
    one = match_1v1(players, "nearest", wait, cfg["matchmaking_config"]["matchmaking"], __import__("numpy").random.default_rng(1))
    assert one
    five = match_5v5(players, "multi_objective", wait, cfg["matchmaking_config"]["matchmaking"], cfg["matchmaking_config"]["multi_objective_weights"], __import__("numpy").random.default_rng(1))
    assert five
    assert len(five[0]["team_a"]) == 5
    assert len(five[0]["team_b"]) == 5
