from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from .player_state import PlayerState
from .skill_model import mmr_win_probability
from .team_builder import assign_team_roles, split_balanced_teams, team_quality


def match_1v1(queue: list[PlayerState], policy: str, wait: dict[int, int], cfg: dict, rng: np.random.Generator) -> list[tuple[PlayerState, PlayerState]]:
    players = list(queue)
    if policy == "random":
        rng.shuffle(players)
    else:
        players = sorted(players, key=lambda p: (p.hidden_mmr, p.player_id))
    matches: list[tuple[PlayerState, PlayerState]] = []
    used: set[int] = set()
    base_window = cfg["base_window"]
    growth = cfg["window_growth_per_wait"]
    for p in players:
        if p.player_id in used:
            continue
        candidates = [q for q in players if q.player_id not in used and q.player_id != p.player_id]
        if not candidates:
            break
        if policy in {"expanding", "latency_constrained"}:
            window = base_window + growth * wait.get(p.player_id, 0)
            candidates = [q for q in candidates if abs(q.hidden_mmr - p.hidden_mmr) <= window] or candidates
        if policy == "fairness_constrained":
            candidates = [
                q for q in candidates
                if cfg["fairness_winprob_low"] <= mmr_win_probability(p.hidden_mmr, q.hidden_mmr) <= cfg["fairness_winprob_high"]
            ] or candidates
        partner = min(candidates, key=lambda q: abs(q.hidden_mmr - p.hidden_mmr))
        matches.append((p, partner))
        used.add(p.player_id)
        used.add(partner.player_id)
    return matches


def _candidate_score(group: Sequence[PlayerState], cfg: dict, weights: dict, wait: dict[int, int]) -> float:
    a, b = split_balanced_teams(group)
    ra = assign_team_roles(a)
    rb = assign_team_roles(b)
    q = team_quality(a, b, ra, rb)
    waiting = np.mean([wait.get(p.player_id, 0) for p in group]) / max(1.0, cfg["max_wait"])
    uncertainty = np.mean([p.rating_uncertainty for p in group]) / 350.0
    new_exposure = sum(p.new_player or p.smurf for p in group) / len(group)
    return (
        weights["fairness_loss"] * q["fairness_loss"]
        + weights["waiting_cost"] * waiting
        + weights["role_penalty"] * q["role_mismatch_rate"]
        + weights["uncertainty_penalty"] * uncertainty
        + weights["new_player_exposure_penalty"] * new_exposure
    )


def _policy_candidate_score(group: Sequence[PlayerState], policy: str, cfg: dict, weights: dict, wait: dict[int, int]) -> float:
    a, b = split_balanced_teams(group)
    ra = assign_team_roles(a)
    rb = assign_team_roles(b)
    q = team_quality(a, b, ra, rb)
    waiting = np.mean([wait.get(p.player_id, 0) for p in group]) / max(1.0, cfg["max_wait"])
    uncertainty = np.mean([p.rating_uncertainty for p in group]) / 350.0
    new_or_smurf = sum(p.new_player or p.smurf for p in group) / len(group)
    smurf_like = sum(p.smurf for p in group) / len(group)
    latent_hidden_mismatch = np.mean([abs((1500.0 + 400.0 * p.latent_global_skill) - p.hidden_mmr) for p in group]) / 600.0
    if policy == "role_aware":
        return (
            1.20 * q["role_mismatch_rate"]
            + 0.18 * q["fairness_loss"]
            + 0.08 * uncertainty
            + 0.04 * waiting
        )
    if policy == "fairness_constrained":
        in_band = cfg["fairness_winprob_low"] <= q["expected_win_prob"] <= cfg["fairness_winprob_high"]
        band_penalty = 0.0 if in_band else 2.0
        return band_penalty + q["fairness_loss"] + 0.10 * q["role_mismatch_rate"] + 0.04 * waiting
    score = (
        weights["fairness_loss"] * q["fairness_loss"]
        + weights["waiting_cost"] * waiting
        + weights["role_penalty"] * q["role_mismatch_rate"]
        + weights["uncertainty_penalty"] * uncertainty
        + weights["new_player_exposure_penalty"] * new_or_smurf
    )
    if policy in {"multi_objective_with_smurf_exposure_penalty", "multi_objective_subgroup_audit"}:
        score += float(weights.get("smurf_exposure_penalty", cfg.get("smurf_separation_penalty", 0.12))) * (
            0.65 * smurf_like + 0.35 * latent_hidden_mismatch
        )
    if policy == "multi_objective_without_smurf_penalty":
        return score
    return score


def match_5v5(
    queue: list[PlayerState],
    policy: str,
    wait: dict[int, int],
    cfg: dict,
    weights: dict,
    rng: np.random.Generator,
) -> list[dict]:
    players = list(queue)
    if len(players) < 10:
        return []
    if policy == "random":
        rng.shuffle(players)
    else:
        players = sorted(players, key=lambda p: (p.hidden_mmr, p.player_id))
    matches: list[dict] = []
    idx = 0
    available = players
    while len(available) >= 10:
        if policy in {
            "role_aware",
            "multi_objective",
            "multi_objective_without_smurf_penalty",
            "multi_objective_with_smurf_exposure_penalty",
            "multi_objective_subgroup_audit",
            "fairness_constrained",
        }:
            anchor = available[0]
            pool = sorted(
                available,
                key=lambda p: (abs(p.hidden_mmr - anchor.hidden_mmr), -wait.get(p.player_id, 0), p.player_id),
            )[: min(int(cfg.get("candidate_pool_size", 30)), len(available))]
            sampled_groups = []
            if len(pool) >= 10:
                for _ in range(24):
                    ids = rng.choice(len(pool), size=10, replace=False)
                    sampled_groups.append([pool[i] for i in ids])
                sampled_groups.append(pool[:10])
            group = min(sampled_groups, key=lambda g: _policy_candidate_score(g, policy, cfg, weights, wait))
        elif policy == "expanding":
            anchor = available[0]
            window = cfg["base_window"] + cfg["window_growth_per_wait"] * wait.get(anchor.player_id, 0)
            group = [p for p in available if abs(p.hidden_mmr - anchor.hidden_mmr) <= window][:10]
            if len(group) < 10:
                group = available[:10]
        else:
            group = available[:10]
        group_ids = {p.player_id for p in group}
        if len(group_ids) < 10:
            break
        team_a, team_b = split_balanced_teams(group)
        roles_a = assign_team_roles(team_a)
        roles_b = assign_team_roles(team_b)
        q = team_quality(team_a, team_b, roles_a, roles_b)
        if policy == "fairness_constrained":
            if not (cfg["fairness_winprob_low"] <= q["expected_win_prob"] <= cfg["fairness_winprob_high"]):
                idx += 1
                available = available[1:]
                if idx > len(players):
                    break
                continue
        matches.append({"team_a": team_a, "team_b": team_b, "roles_a": roles_a, "roles_b": roles_b, "quality": q})
        available = [p for p in available if p.player_id not in group_ids]
        idx += 1
    return matches
