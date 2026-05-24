from matchmaking_sim.config import load_all_configs
from matchmaking_sim.population import generate_population


def test_population_contains_required_behavior_types():
    cfg = load_all_configs()
    players = generate_population(120, cfg["player_population_config"], cfg["rating_config"], 7)
    assert len(players) == 120
    assert all(p.hidden_mmr > 0 for p in players)
    assert {p.behavior_type for p in players}
    assert all(p.preferred_role for p in players)
