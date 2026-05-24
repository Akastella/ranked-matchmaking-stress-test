from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi, sqrt


GLICKO2_SCALE = 173.7178


@dataclass
class Glicko2State:
    rating: float = 1500.0
    rd: float = 350.0
    volatility: float = 0.06


def _g(phi: float) -> float:
    return 1.0 / sqrt(1.0 + 3.0 * phi * phi / (pi * pi))


def _e(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + exp(-_g(phi_j) * (mu - mu_j)))


def glicko2_single_update(
    player: Glicko2State,
    opponent: Glicko2State,
    score: float,
    tau: float = 0.5,
    epsilon: float = 1e-6,
) -> Glicko2State:
    """Standard Glicko-2 one-opponent update.

    This is used only as a sanity baseline. It follows the public Glicko-2
    update equations for a single rating period with one opponent; it is not
    used as the production mechanism in the simulation framework.
    """
    mu = (player.rating - 1500.0) / GLICKO2_SCALE
    phi = player.rd / GLICKO2_SCALE
    sigma = player.volatility
    mu_j = (opponent.rating - 1500.0) / GLICKO2_SCALE
    phi_j = opponent.rd / GLICKO2_SCALE

    e_val = _e(mu, mu_j, phi_j)
    g_val = _g(phi_j)
    v = 1.0 / (g_val * g_val * e_val * (1.0 - e_val))
    delta = v * g_val * (score - e_val)

    a = log(sigma * sigma)

    def f(x: float) -> float:
        ex = exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (tau * tau)

    A = a
    if delta * delta > phi * phi + v:
        B = log(delta * delta - phi * phi - v)
    else:
        k = 1
        B = a - k * tau
        while f(B) < 0:
            k += 1
            B = a - k * tau

    f_a = f(A)
    f_b = f(B)
    while abs(B - A) > epsilon:
        C = A + (A - B) * f_a / (f_b - f_a)
        f_c = f(C)
        if f_c * f_b <= 0:
            A = B
            f_a = f_b
        else:
            f_a = f_a / 2.0
        B = C
        f_b = f_c

    sigma_prime = exp(A / 2.0)
    phi_star = sqrt(phi * phi + sigma_prime * sigma_prime)
    phi_prime = 1.0 / sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_prime = mu + phi_prime * phi_prime * g_val * (score - e_val)

    return Glicko2State(
        rating=1500.0 + GLICKO2_SCALE * mu_prime,
        rd=GLICKO2_SCALE * phi_prime,
        volatility=sigma_prime,
    )


def sanity_check_series() -> list[dict]:
    player = Glicko2State()
    stronger = Glicko2State(rating=1700.0, rd=80.0, volatility=0.06)
    weaker = Glicko2State(rating=1300.0, rd=80.0, volatility=0.06)
    rows = []
    for label, opponent, score in [
        ("win_vs_stronger", stronger, 1.0),
        ("loss_vs_weaker", weaker, 0.0),
        ("draw_vs_equal", Glicko2State(rating=1500.0, rd=120.0), 0.5),
    ]:
        updated = glicko2_single_update(player, opponent, score)
        rows.append(
            {
                "case": label,
                "initial_rating": player.rating,
                "opponent_rating": opponent.rating,
                "score": score,
                "updated_rating": updated.rating,
                "updated_rd": updated.rd,
                "updated_volatility": updated.volatility,
                "rating_delta": updated.rating - player.rating,
            }
        )
    return rows
