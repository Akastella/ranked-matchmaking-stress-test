from __future__ import annotations

from collections.abc import Sequence

from .player_state import PlayerState
from .rank_presentation import update_visible_rank
from .skill_model import mmr_win_probability


class RatingSystem:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg["rating"]
        self.full_cfg = cfg

    def expected(self, a: PlayerState, b: PlayerState) -> float:
        return mmr_win_probability(a.hidden_mmr, b.hidden_mmr, self.cfg["scale"])

    def update_1v1(self, a: PlayerState, b: PlayerState, outcome_a: int, role_a="flex", role_b="flex") -> None:
        raise NotImplementedError

    def update_team(
        self,
        team_a: Sequence[PlayerState],
        team_b: Sequence[PlayerState],
        roles_a: Sequence[str],
        roles_b: Sequence[str],
        outcome_a: int,
    ) -> None:
        mean_a = sum(p.hidden_mmr for p in team_a) / len(team_a)
        mean_b = sum(p.hidden_mmr for p in team_b) / len(team_b)
        exp_a = mmr_win_probability(mean_a, mean_b, self.cfg["scale"])
        for p, role in zip(team_a, roles_a):
            self._apply_delta(p, self._k(p) * (outcome_a - exp_a), role)
        for p, role in zip(team_b, roles_b):
            self._apply_delta(p, self._k(p) * ((1 - outcome_a) - (1 - exp_a)), role)

    def inactive_day(self, p: PlayerState) -> None:
        p.rating_uncertainty = min(self.cfg["uncertainty_max"], p.rating_uncertainty + self.cfg["uncertainty_growth_inactive"])

    def _k(self, p: PlayerState) -> float:
        return float(self.cfg["base_k"])

    def _apply_delta(self, p: PlayerState, delta: float, role: str = "flex", context: str = "default") -> None:
        p.hidden_mmr += delta
        p.role_rating[role] = p.role_rating.get(role, p.hidden_mmr) + delta * float(self.cfg["context_weight"])
        p.context_rating[context] = p.context_rating.get(context, p.hidden_mmr) + delta * 0.10
        p.rating_uncertainty = max(self.cfg["uncertainty_min"], p.rating_uncertainty * self.cfg["uncertainty_shrink"])
        p.volatility = 0.85 * p.volatility + 0.15 * abs(delta) / 100.0
        p.match_count += 1
        update_visible_rank(p, self.full_cfg)


class StandardElo(RatingSystem):
    name = "elo"

    def update_1v1(self, a, b, outcome_a, role_a="flex", role_b="flex") -> None:
        exp_a = self.expected(a, b)
        delta = self._k(a) * (outcome_a - exp_a)
        self._apply_delta(a, delta, role_a)
        self._apply_delta(b, -delta, role_b)


class DynamicKElo(StandardElo):
    name = "dynamic_k"

    def _k(self, p: PlayerState) -> float:
        uncertainty_factor = p.rating_uncertainty / self.cfg["uncertainty_initial"]
        experience_factor = 1.0 / (1.0 + p.match_count / 35.0)
        volatility_factor = 1.0 + min(0.8, p.volatility)
        k = self.cfg["base_k"] * (0.55 + uncertainty_factor + experience_factor * 0.35) * volatility_factor
        return float(max(self.cfg["dynamic_k_min"], min(self.cfg["dynamic_k_max"], k)))


class GlickoLike(StandardElo):
    name = "glicko_like"

    def _k(self, p: PlayerState) -> float:
        rd_factor = max(0.25, min(1.8, p.rating_uncertainty / 160.0))
        return float(max(self.cfg["dynamic_k_min"], min(self.cfg["dynamic_k_max"], self.cfg["base_k"] * rd_factor)))

    def inactive_day(self, p: PlayerState) -> None:
        p.rating_uncertainty = min(self.cfg["uncertainty_max"], p.rating_uncertainty + self.cfg["uncertainty_growth_inactive"] * 1.6)


class TrueSkillApprox(DynamicKElo):
    name = "trueskill_approx"

    def update_team(self, team_a, team_b, roles_a, roles_b, outcome_a) -> None:
        mean_a = sum(p.hidden_mmr for p in team_a) / len(team_a)
        mean_b = sum(p.hidden_mmr for p in team_b) / len(team_b)
        var_a = sum(p.rating_uncertainty**2 for p in team_a) / len(team_a)
        var_b = sum(p.rating_uncertainty**2 for p in team_b) / len(team_b)
        exp_a = mmr_win_probability(mean_a, mean_b, self.cfg["scale"])
        team_uncertainty = (var_a + var_b) ** 0.5 / self.cfg["uncertainty_initial"]
        for p, role in zip(team_a, roles_a):
            share = p.rating_uncertainty / max(1.0, sum(x.rating_uncertainty for x in team_a))
            delta = self.cfg["base_k"] * team_uncertainty * share * len(team_a) * (outcome_a - exp_a)
            self._apply_delta(p, delta, role)
        for p, role in zip(team_b, roles_b):
            share = p.rating_uncertainty / max(1.0, sum(x.rating_uncertainty for x in team_b))
            delta = self.cfg["base_k"] * team_uncertainty * share * len(team_b) * ((1 - outcome_a) - (1 - exp_a))
            self._apply_delta(p, delta, role)


class DriftAware(DynamicKElo):
    name = "drift_aware"

    def inactive_day(self, p: PlayerState) -> None:
        p.hidden_mmr = self.cfg["initial_rating"] + self.cfg["drift_decay"] * (p.hidden_mmr - self.cfg["initial_rating"])
        p.rating_uncertainty = min(self.cfg["uncertainty_max"], p.rating_uncertainty + self.cfg["uncertainty_growth_inactive"] * 1.4)


class ContextAware(DynamicKElo):
    name = "context_aware"

    def expected_context(self, a: PlayerState, b: PlayerState, role_a: str, role_b: str, context: str) -> float:
        ar = 0.75 * a.hidden_mmr + 0.20 * a.role_rating.get(role_a, a.hidden_mmr) + 0.05 * a.context_rating.get(context, a.hidden_mmr)
        br = 0.75 * b.hidden_mmr + 0.20 * b.role_rating.get(role_b, b.hidden_mmr) + 0.05 * b.context_rating.get(context, b.hidden_mmr)
        return mmr_win_probability(ar, br, self.cfg["scale"])

    def update_1v1(self, a, b, outcome_a, role_a="flex", role_b="flex") -> None:
        exp_a = self.expected_context(a, b, role_a, role_b, "default")
        delta = self._k(a) * (outcome_a - exp_a)
        self._apply_delta(a, delta, role_a)
        self._apply_delta(b, -delta, role_b)


def make_rating_system(name: str, cfg: dict) -> RatingSystem:
    mapping = {
        "elo": StandardElo,
        "dynamic_k": DynamicKElo,
        "glicko_like": GlickoLike,
        "trueskill_approx": TrueSkillApprox,
        "drift_aware": DriftAware,
        "context_aware": ContextAware,
    }
    return mapping[name](cfg)
