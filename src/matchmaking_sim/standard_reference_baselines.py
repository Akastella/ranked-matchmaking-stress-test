from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from random import Random

from .standard_baselines import Glicko2State, glicko2_single_update


def _prob(a: float, b: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((b - a) / scale))


def _clip_prob(p: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, p))


@dataclass
class SimpleRatingState:
    rating: float
    uncertainty: float = 220.0
    matches: int = 0


def _update_elo(a: SimpleRatingState, b: SimpleRatingState, outcome: float, k: float) -> float:
    expected = _prob(a.rating, b.rating)
    delta = k * (outcome - expected)
    a.rating += delta
    b.rating -= delta
    a.matches += 1
    b.matches += 1
    a.uncertainty = max(60.0, a.uncertainty * 0.985)
    b.uncertainty = max(60.0, b.uncertainty * 0.985)
    return expected


def _dynamic_k(state: SimpleRatingState) -> float:
    uncertainty_factor = state.uncertainty / 220.0
    experience_factor = 1.0 / (1.0 + state.matches / 35.0)
    return max(12.0, min(48.0, 22.0 * (0.65 + uncertainty_factor + 0.25 * experience_factor)))


def run_reference_rating_check(seed: int, n_players: int = 360, n_matches: int = 4200) -> list[dict]:
    rng = Random(seed)
    latent = [rng.gauss(1500.0, 260.0) for _ in range(n_players)]
    initial = [skill + rng.gauss(0.0, 230.0) for skill in latent]

    systems = {
        "elo": [SimpleRatingState(r) for r in initial],
        "dynamic_k": [SimpleRatingState(r) for r in initial],
        "glicko_like_internal": [SimpleRatingState(r) for r in initial],
        "standard_glicko2_reference": [Glicko2State(rating=r, rd=220.0, volatility=0.06) for r in initial],
    }
    metrics = {name: {"brier": [], "log_loss": []} for name in systems}

    for _ in range(n_matches):
        a_idx, b_idx = rng.sample(range(n_players), 2)
        true_p = _prob(latent[a_idx], latent[b_idx])
        outcome = 1.0 if rng.random() < true_p else 0.0

        exp_elo = _update_elo(systems["elo"][a_idx], systems["elo"][b_idx], outcome, 24.0)
        metrics["elo"]["brier"].append((outcome - exp_elo) ** 2)
        metrics["elo"]["log_loss"].append(-(outcome * log(_clip_prob(exp_elo)) + (1 - outcome) * log(_clip_prob(1 - exp_elo))))

        k_a = _dynamic_k(systems["dynamic_k"][a_idx])
        k_b = _dynamic_k(systems["dynamic_k"][b_idx])
        exp_dyn = _prob(systems["dynamic_k"][a_idx].rating, systems["dynamic_k"][b_idx].rating)
        delta = (0.5 * (k_a + k_b)) * (outcome - exp_dyn)
        systems["dynamic_k"][a_idx].rating += delta
        systems["dynamic_k"][b_idx].rating -= delta
        for state in (systems["dynamic_k"][a_idx], systems["dynamic_k"][b_idx]):
            state.matches += 1
            state.uncertainty = max(60.0, state.uncertainty * 0.98)
        metrics["dynamic_k"]["brier"].append((outcome - exp_dyn) ** 2)
        metrics["dynamic_k"]["log_loss"].append(-(outcome * log(_clip_prob(exp_dyn)) + (1 - outcome) * log(_clip_prob(1 - exp_dyn))))

        g_a = systems["glicko_like_internal"][a_idx]
        g_b = systems["glicko_like_internal"][b_idx]
        k_g = max(12.0, min(48.0, 24.0 * max(0.25, min(1.8, (g_a.uncertainty + g_b.uncertainty) / 320.0))))
        exp_g = _update_elo(g_a, g_b, outcome, k_g)
        metrics["glicko_like_internal"]["brier"].append((outcome - exp_g) ** 2)
        metrics["glicko_like_internal"]["log_loss"].append(-(outcome * log(_clip_prob(exp_g)) + (1 - outcome) * log(_clip_prob(1 - exp_g))))

        ga = systems["standard_glicko2_reference"][a_idx]
        gb = systems["standard_glicko2_reference"][b_idx]
        exp_std = _prob(ga.rating, gb.rating)
        systems["standard_glicko2_reference"][a_idx] = glicko2_single_update(ga, gb, outcome)
        systems["standard_glicko2_reference"][b_idx] = glicko2_single_update(gb, ga, 1.0 - outcome)
        metrics["standard_glicko2_reference"]["brier"].append((outcome - exp_std) ** 2)
        metrics["standard_glicko2_reference"]["log_loss"].append(-(outcome * log(_clip_prob(exp_std)) + (1 - outcome) * log(_clip_prob(1 - exp_std))))

    rows: list[dict] = []
    for name, states in systems.items():
        ratings = [s.rating for s in states]
        errors = [abs(r - t) for r, t in zip(ratings, latent)]
        rmse = (sum((r - t) ** 2 for r, t in zip(ratings, latent)) / len(latent)) ** 0.5
        rows.append(
            {
                "seed": seed,
                "baseline": name,
                "n_players": n_players,
                "n_matches": n_matches,
                "rating_mae": sum(errors) / len(errors),
                "rating_rmse": rmse,
                "brier_score": sum(metrics[name]["brier"]) / len(metrics[name]["brier"]),
                "log_loss": sum(metrics[name]["log_loss"]) / len(metrics[name]["log_loss"]),
            }
        )
    return rows
