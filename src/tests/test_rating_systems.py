from matchmaking_sim.config import load_all_configs
from matchmaking_sim.population import generate_population
from matchmaking_sim.rating_systems import make_rating_system


def test_rating_update_changes_hidden_mmr_and_visible_rank():
    cfg = load_all_configs()
    players = generate_population(2, cfg["player_population_config"], cfg["rating_config"], 3)
    a, b = players
    old = a.hidden_mmr
    system = make_rating_system("dynamic_k", cfg["rating_config"])
    system.update_1v1(a, b, 1, a.preferred_role, b.preferred_role)
    assert a.hidden_mmr != old
    assert a.visible_rank
    assert a.match_count == 1
