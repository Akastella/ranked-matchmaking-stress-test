from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .player_state import ROLES, PlayerState


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def win_probability(skill_a: float, skill_b: float) -> float:
    return logistic(skill_a - skill_b)


def mmr_win_probability(mmr_a: float, mmr_b: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10 ** ((mmr_b - mmr_a) / scale))


def team_strength(team: Sequence[PlayerState], roles: Sequence[str], context: str, sim_cfg: dict) -> float:
    if not team:
        return 0.0
    skills = np.array([p.effective_skill(role, context) for p, role in zip(team, roles)], dtype=float)
    role_coverage = len(set(roles)) / max(1, len(ROLES))
    coverage_penalty = (1.0 - role_coverage) * float(sim_cfg["role_mismatch_strength_penalty"])
    variance_penalty = float(np.var(skills)) * float(sim_cfg["within_team_variance_penalty"])
    return float(np.mean(skills) - coverage_penalty - variance_penalty)


def generate_outcome(prob_a: float, rng: np.random.Generator) -> int:
    return int(rng.random() < prob_a)
