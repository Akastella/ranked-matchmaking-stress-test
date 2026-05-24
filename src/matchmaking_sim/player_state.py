from __future__ import annotations

from dataclasses import dataclass, field


ROLES = ["duelist", "controller", "initiator", "sentinel", "flex"]
CONTEXTS = ["default", "map_a", "map_b", "map_c"]


@dataclass
class PlayerState:
    player_id: int
    latent_global_skill: float
    latent_role_skill: dict[str, float]
    latent_context_skill: dict[str, float]
    hidden_mmr: float
    visible_rank_points: float
    visible_rank: str
    rating_uncertainty: float
    volatility: float
    activity_rate: float
    recent_form: float
    experience_level: float
    new_player: bool
    smurf: bool
    churn_risk_state: float
    preferred_role: str
    secondary_role: str
    party_id: int | None
    behavior_type: str
    match_count: int = 0
    wins: int = 0
    losses: int = 0
    loss_streak: int = 0
    unfair_match_count: int = 0
    repeated_large_skill_gap_exposure: int = 0
    smurf_victimization: int = 0
    waiting_time_frustration: float = 0.0
    rank_mmr_divergence: float = 0.0
    last_active_day: int = 0
    role_rating: dict[str, float] = field(default_factory=dict)
    context_rating: dict[str, float] = field(default_factory=dict)

    @property
    def is_placement(self) -> bool:
        return self.match_count < 10 or self.new_player

    def role_skill(self, role: str) -> float:
        return self.latent_global_skill + self.latent_role_skill.get(role, 0.0)

    def context_skill(self, context: str) -> float:
        return self.latent_global_skill + self.latent_context_skill.get(context, 0.0)

    def effective_skill(self, role: str, context: str = "default") -> float:
        role_component = self.latent_role_skill.get(role, 0.0)
        context_component = self.latent_context_skill.get(context, 0.0)
        return self.latent_global_skill + role_component + context_component + self.recent_form
