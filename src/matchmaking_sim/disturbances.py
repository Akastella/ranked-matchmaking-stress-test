from __future__ import annotations

import numpy as np

from .rank_presentation import seasonal_compress


def apply_daily_drift(players, sim_cfg: dict, rng: np.random.Generator) -> None:
    for p in players:
        drift = rng.normal(0, sim_cfg["skill_drift_sd"])
        if p.behavior_type == "improving_player":
            drift += sim_cfg["improving_drift"]
        elif p.behavior_type == "declining_player":
            drift += sim_cfg["declining_drift"]
        elif p.behavior_type == "unstable_player":
            drift += rng.normal(0, sim_cfg["unstable_drift_sd"])
        p.latent_global_skill += float(drift)
        p.recent_form = sim_cfg["recent_form_decay"] * p.recent_form + float(rng.normal(0, sim_cfg["recent_form_noise"]))


def apply_seasonal_reset(players, day: int, sim_cfg: dict, rating_cfg: dict) -> bool:
    if day == int(sim_cfg["seasonal_reset_day"]):
        for p in players:
            seasonal_compress(p, rating_cfg)
        return True
    return False
