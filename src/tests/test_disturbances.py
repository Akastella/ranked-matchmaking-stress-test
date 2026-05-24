from matchmaking_sim.config import load_all_configs
from matchmaking_sim.disturbances import apply_daily_drift
from matchmaking_sim.population import generate_population
import numpy as np


def test_drift_changes_latent_skill():
    cfg = load_all_configs()
    players = generate_population(10, cfg["player_population_config"], cfg["rating_config"], 8)
    old = [p.latent_global_skill for p in players]
    apply_daily_drift(players, cfg["simulation_config"]["simulation"], np.random.default_rng(8))
    assert any(a != b for a, b in zip(old, [p.latent_global_skill for p in players]))
