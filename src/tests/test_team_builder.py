from matchmaking_sim.config import load_all_configs
from matchmaking_sim.population import generate_population
from matchmaking_sim.team_builder import assign_team_roles, split_balanced_teams, team_quality


def test_team_quality_has_expected_keys():
    cfg = load_all_configs()
    players = generate_population(10, cfg["player_population_config"], cfg["rating_config"], 6)
    a, b = split_balanced_teams(players)
    qa = team_quality(a, b, assign_team_roles(a), assign_team_roles(b))
    assert 0 <= qa["expected_win_prob"] <= 1
    assert qa["team_mmr_gap"] >= 0
