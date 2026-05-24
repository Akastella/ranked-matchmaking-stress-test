from __future__ import annotations

import numpy as np

from .skill_model import generate_outcome, team_strength, win_probability


def outcome_1v1(a, b, role_a, role_b, context, rng: np.random.Generator) -> tuple[int, float]:
    prob = win_probability(a.effective_skill(role_a, context), b.effective_skill(role_b, context))
    return generate_outcome(prob, rng), prob


def outcome_team(team_a, team_b, roles_a, roles_b, context, sim_cfg, rng: np.random.Generator) -> tuple[int, float]:
    sa = team_strength(team_a, roles_a, context, sim_cfg)
    sb = team_strength(team_b, roles_b, context, sim_cfg)
    prob = win_probability(sa, sb)
    return generate_outcome(prob, rng), prob
