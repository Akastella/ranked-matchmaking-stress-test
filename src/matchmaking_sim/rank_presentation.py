from __future__ import annotations


def rank_name(points: float, rank_cfg: dict) -> str:
    names = rank_cfg["names"]
    floor = float(rank_cfg["floor"])
    step = float(rank_cfg["step"])
    idx = int(max(0, min(len(names) - 1, (points - floor) // step)))
    return names[idx]


def update_visible_rank(player, rating_cfg: dict) -> None:
    params = rating_cfg.get("rating", rating_cfg)
    smoothing = float(params["visible_rank_smoothing"])
    player.visible_rank_points += smoothing * (player.hidden_mmr - player.visible_rank_points)
    player.visible_rank = rank_name(player.visible_rank_points, rating_cfg["ranks"])
    player.rank_mmr_divergence = abs(player.hidden_mmr - player.visible_rank_points)


def seasonal_compress(player, rating_cfg: dict) -> None:
    params = rating_cfg["rating"]
    center = float(params["initial_rating"])
    compression = float(params["seasonal_compression"])
    player.hidden_mmr = center + compression * (player.hidden_mmr - center)
    player.visible_rank_points = center + compression * (player.visible_rank_points - center)
    player.rating_uncertainty = min(
        float(params["uncertainty_max"]),
        player.rating_uncertainty + 45,
    )
    update_visible_rank(player, rating_cfg)
