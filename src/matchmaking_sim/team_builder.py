from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .player_state import ROLES, PlayerState
from .skill_model import mmr_win_probability


def assign_team_roles(team: Sequence[PlayerState]) -> list[str]:
    assigned: list[str] = []
    remaining = list(ROLES)
    for p in team:
        if p.preferred_role in remaining:
            role = p.preferred_role
        elif p.secondary_role in remaining:
            role = p.secondary_role
        elif remaining:
            role = remaining[0]
        else:
            role = p.preferred_role
        assigned.append(role)
        if role in remaining:
            remaining.remove(role)
    return assigned


def split_balanced_teams(players: Sequence[PlayerState]) -> tuple[list[PlayerState], list[PlayerState]]:
    ordered = sorted(players, key=lambda p: p.hidden_mmr, reverse=True)
    a: list[PlayerState] = []
    b: list[PlayerState] = []
    sum_a = 0.0
    sum_b = 0.0
    for p in ordered:
        if sum_a <= sum_b:
            a.append(p)
            sum_a += p.hidden_mmr
        else:
            b.append(p)
            sum_b += p.hidden_mmr
    return a, b


def role_mismatch_rate(team: Sequence[PlayerState], roles: Sequence[str]) -> float:
    coverage_failure = 1.0 - (len(set(roles)) / max(1, len(ROLES)))
    off_role = np.mean(
        [
            0.0 if assigned in {p.preferred_role, p.secondary_role} else 1.0
            for p, assigned in zip(team, roles, strict=False)
        ]
    )
    return float(0.5 * coverage_failure + 0.5 * off_role)


def team_quality(team_a: Sequence[PlayerState], team_b: Sequence[PlayerState], roles_a: Sequence[str], roles_b: Sequence[str]) -> dict[str, float]:
    mmr_a = float(np.mean([p.hidden_mmr for p in team_a]))
    mmr_b = float(np.mean([p.hidden_mmr for p in team_b]))
    expected = mmr_win_probability(mmr_a, mmr_b)
    return {
        "team_mmr_gap": abs(mmr_a - mmr_b),
        "expected_win_prob": expected,
        "fairness_loss": abs(expected - 0.5),
        "role_mismatch_rate": (role_mismatch_rate(team_a, roles_a) + role_mismatch_rate(team_b, roles_b)) / 2.0,
        "within_team_variance": float((np.var([p.hidden_mmr for p in team_a]) + np.var([p.hidden_mmr for p in team_b])) / 2.0),
    }
