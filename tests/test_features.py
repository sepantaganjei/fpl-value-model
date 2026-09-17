"""Tests for :mod:`fpl_value_model.features`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_value_model.config import FEATURE_COLUMNS, PER90_FEATURES
from fpl_value_model.features import (
    add_price_m,
    attach_history_features,
    build_feature_frame,
    latest_history_per_player,
    to_per90,
)


def test_add_price_m_divides_by_ten(raw_history: pd.DataFrame) -> None:
    out = add_price_m(raw_history)
    assert (out["price_m"] == raw_history["now_cost"] / 10).all()


def test_to_per90_matches_manual_rate() -> None:
    frame = pd.DataFrame(
        {
            "minutes": [90, 180, 0],
            "goals_scored": [1, 1, 5],
            "assists": [0, 2, 0],
            "expected_goals": [0.5, 1.0, 0.0],
            "expected_assists": [0.1, 0.2, 0.0],
            "expected_goal_involvements": [0.6, 1.2, 0.0],
            "ict_index": [10.0, 20.0, 0.0],
            "bonus": [3, 3, 0],
            "clean_sheets": [1, 1, 0],
            "saves": [0, 0, 0],
        }
    )
    out = to_per90(frame)
    assert out.loc[0, "goals_scored_per90"] == 1.0
    assert out.loc[1, "goals_scored_per90"] == 0.5
    # Zero minutes must not produce inf/NaN.
    assert out.loc[2, "goals_scored_per90"] == 0.0


def test_build_feature_frame_applies_minutes_threshold(
    raw_history: pd.DataFrame,
) -> None:
    out = build_feature_frame(raw_history, min_minutes=1000)
    assert (out["minutes"] >= 1000).all()
    assert set(FEATURE_COLUMNS).issubset(out.columns)
    assert "price_m" in out.columns


def test_build_feature_frame_has_no_missing_or_infinite(
    raw_history: pd.DataFrame,
) -> None:
    out = build_feature_frame(raw_history)
    numeric = out[list(PER90_FEATURES)].to_numpy()
    assert np.isfinite(numeric).all()
    assert not out[list(FEATURE_COLUMNS)].isna().any().any()


def test_latest_history_per_player_keeps_most_recent(
    raw_history: pd.DataFrame,
) -> None:
    out = latest_history_per_player(build_feature_frame(raw_history))
    assert out["name"].is_unique
    assert (out["season"] == "2024-25").any()
    assert not (out["season"] == "2022-23").any()


def test_attach_history_features_inner_joins_on_name(
    raw_history: pd.DataFrame, raw_live: pd.DataFrame
) -> None:
    out = attach_history_features(raw_live, raw_history)
    assert len(out) > 0
    assert len(out) <= len(raw_live)
    assert set(FEATURE_COLUMNS).issubset(out.columns)
    # price_m comes from the live pool, not history.
    merged = out.merge(add_price_m(raw_live)[["name", "price_m"]], on="name")
    assert np.allclose(merged["price_m_x"], merged["price_m_y"])


def test_attach_history_features_drops_players_without_history(
    raw_history: pd.DataFrame, raw_live: pd.DataFrame
) -> None:
    newcomer = raw_live.iloc[[0]].copy()
    newcomer["name"] = "Brand New Signing"
    live = pd.concat([raw_live, newcomer], ignore_index=True)
    out = attach_history_features(live, raw_history)
    assert "Brand New Signing" not in set(out["name"])
