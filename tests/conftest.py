"""Shared fixtures: small synthetic player-season frames on the real schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_COLUMNS = [
    "player_id",
    "name",
    "position",
    "team_id",
    "now_cost",
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
    "bonus",
    "clean_sheets",
    "saves",
    "selected_by_percent",
    "form",
    "season",
]


def _player_season(
    name: str,
    season: str,
    *,
    position: str = "MID",
    minutes: int = 2500,
    goals: int = 6,
    assists: int = 5,
    now_cost: int = 70,
    points: int = 150,
) -> dict[str, object]:
    xg = goals * 0.9
    xa = assists * 0.8
    return {
        "player_id": abs(hash((name, season))) % 1000,
        "name": name,
        "position": position,
        "team_id": 1,
        "now_cost": now_cost,
        "total_points": points,
        "minutes": minutes,
        "goals_scored": goals,
        "assists": assists,
        "expected_goals": xg,
        "expected_assists": xa,
        "expected_goal_involvements": xg + xa,
        "ict_index": (goals + assists) * 12.0,
        "bonus": points // 12,
        "clean_sheets": 8 if position in {"GK", "DEF"} else 2,
        "saves": 90 if position == "GK" else 0,
        "selected_by_percent": 10.0,
        "form": round(points / 38, 1),
        "season": season,
    }


@pytest.fixture
def raw_history() -> pd.DataFrame:
    """Three seasons of ~40 players each, priced roughly on output."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    positions = ["GK", "DEF", "DEF", "MID", "MID", "FWD"]
    for season in ("2022-23", "2023-24", "2024-25"):
        for i in range(40):
            position = positions[i % len(positions)]
            minutes = int(rng.integers(300, 3200))
            goals = int(rng.integers(0, 20)) if position != "GK" else 0
            assists = int(rng.integers(0, 12)) if position != "GK" else 0
            points = 40 + goals * 5 + assists * 3 + int(rng.integers(0, 60))
            now_cost = 40 + goals * 2 + assists + int(rng.integers(0, 15))
            rows.append(
                _player_season(
                    f"Player {i:02d}",
                    season,
                    position=position,
                    minutes=minutes,
                    goals=goals,
                    assists=assists,
                    now_cost=now_cost,
                    points=points,
                )
            )
    return pd.DataFrame(rows, columns=_COLUMNS)


@pytest.fixture
def raw_live(raw_history: pd.DataFrame) -> pd.DataFrame:
    """Build a live pool: history names, low minutes, ``season == 'live'``."""
    latest = raw_history[raw_history["season"] == "2024-25"].copy()
    latest["season"] = "live"
    latest["minutes"] = 90
    latest["total_points"] = 3
    latest["now_cost"] = latest["now_cost"] + 2
    return latest.reset_index(drop=True)
