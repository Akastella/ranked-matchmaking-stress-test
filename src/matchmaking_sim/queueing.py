from __future__ import annotations

import numpy as np

from .player_state import PlayerState


def active_players(players: list[PlayerState], rng: np.random.Generator, pressure: float = 1.0) -> list[PlayerState]:
    active = [p for p in players if rng.random() < min(0.99, p.activity_rate * pressure)]
    return sorted(active, key=lambda p: (p.hidden_mmr, p.player_id))


def assign_wait_times(players: list[PlayerState], rng: np.random.Generator, max_wait: int) -> dict[int, int]:
    return {p.player_id: int(rng.integers(0, max_wait + 1)) for p in players}
