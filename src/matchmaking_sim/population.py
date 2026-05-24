from __future__ import annotations

import numpy as np

from .player_state import CONTEXTS, ROLES, PlayerState
from .rank_presentation import rank_name


BEHAVIORS = [
    "regular",
    "new_player",
    "improving_player",
    "declining_player",
    "smurf",
    "unstable_player",
    "low_activity_player",
]


def _choose_behavior(i: int, n: int, cfg: dict, rng: np.random.Generator) -> tuple[str, bool, bool]:
    pop = cfg["population"]
    r = rng.random()
    thresholds = [
        ("smurf", pop["smurf_ratio"]),
        ("new_player", pop["new_player_ratio"]),
        ("improving_player", pop["improving_ratio"]),
        ("declining_player", pop["declining_ratio"]),
        ("unstable_player", pop["unstable_ratio"]),
        ("low_activity_player", pop["low_activity_ratio"]),
    ]
    acc = 0.0
    for behavior, prob in thresholds:
        acc += float(prob)
        if r < acc:
            return behavior, behavior == "new_player", behavior == "smurf"
    return "regular", False, False


def generate_population(n_players: int, cfg: dict, rating_cfg: dict, seed: int) -> list[PlayerState]:
    rng = np.random.default_rng(seed)
    pop = cfg["population"]
    rank_cfg = rating_cfg["ranks"]
    players: list[PlayerState] = []
    for pid in range(n_players):
        behavior, new_player, smurf = _choose_behavior(pid, n_players, cfg, rng)
        latent = float(rng.normal(pop["latent_skill_mean"], pop["latent_skill_sd"]))
        initial_mmr = float(rng.normal(pop["initial_mmr_mean"], pop["initial_mmr_sd"]))
        if smurf:
            latent_gap = float(pop.get("smurf_latent_skill_boost", rng.uniform(1.2, 2.2)))
            mmr_gap_min = float(pop.get("smurf_initial_mmr_gap_min", 250))
            mmr_gap_max = float(pop.get("smurf_initial_mmr_gap_max", 520))
            latent += latent_gap
            initial_mmr -= float(rng.uniform(mmr_gap_min, mmr_gap_max))
        if new_player:
            initial_mmr = float(pop["initial_mmr_mean"] + rng.normal(0, 80))
        role_skill = {role: float(rng.normal(0, pop["role_skill_sd"])) for role in ROLES}
        context_skill = {ctx: float(rng.normal(0, pop["context_skill_sd"])) for ctx in CONTEXTS}
        preferred = str(rng.choice(ROLES, p=[0.30, 0.18, 0.18, 0.16, 0.18]))
        secondary = str(rng.choice([r for r in ROLES if r != preferred]))
        activity = float(np.clip(rng.normal(pop["activity_rate_mean"], pop["activity_rate_sd"]), 0.04, 0.98))
        if behavior == "low_activity_player":
            activity = float(np.clip(activity * 0.35, 0.02, 0.35))
        p = PlayerState(
            player_id=pid,
            latent_global_skill=latent,
            latent_role_skill=role_skill,
            latent_context_skill=context_skill,
            hidden_mmr=initial_mmr,
            visible_rank_points=initial_mmr,
            visible_rank=rank_name(initial_mmr, rank_cfg),
            rating_uncertainty=float(rating_cfg["rating"]["uncertainty_initial"]),
            volatility=float(rating_cfg["rating"]["volatility_initial"]),
            activity_rate=activity,
            recent_form=float(rng.normal(0, 0.05)),
            experience_level=float(rng.uniform(0, 1)),
            new_player=new_player,
            smurf=smurf,
            churn_risk_state=0.0,
            preferred_role=preferred,
            secondary_role=secondary,
            party_id=None,
            behavior_type=behavior,
            role_rating={role: initial_mmr for role in ROLES},
            context_rating={ctx: initial_mmr for ctx in CONTEXTS},
        )
        players.append(p)
    return players
