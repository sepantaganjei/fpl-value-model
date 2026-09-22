"""Tests for :mod:`fpl_value_model.features`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_attach_history_features_keeps_every_live_player(
    raw_history: pd.DataFrame, raw_live: pd.DataFrame
) -> None:
    out = attach_history_features(raw_live, raw_history)
    assert len(out) == len(raw_live)
    assert set(FEATURE_COLUMNS).issubset(out.columns)
    # price_m comes from the live pool, not history.
    merged = out.merge(add_price_m(raw_live)[["name", "price_m"]], on="name")
    assert np.allclose(merged["price_m_x"], merged["price_m_y"])


def test_attach_history_features_falls_back_to_position_average(
    raw_history: pd.DataFrame, raw_live: pd.DataFrame
) -> None:
    newcomer = raw_live.iloc[[0]].copy()
    newcomer["name"] = "Brand New Signing"
    newcomer["position"] = "FWD"
    newcomer["minutes"] = 0
    live = pd.concat([raw_live, newcomer], ignore_index=True)
    out = attach_history_features(live, raw_history)

    row = out.loc[out["name"] == "Brand New Signing"]
    assert len(row) == 1
    assert not row["has_history"].iloc[0]

    fwd_hist = latest_history_per_player(
        build_feature_frame(raw_history, require_target=False)
    )
    fwd_hist = fwd_hist.loc[fwd_hist["position"] == "FWD"]
    assert row["goals_scored_per90"].iloc[0] == pytest.approx(
        fwd_hist["goals_scored_per90"].mean()
    )

    # Players with their own history are unaffected.
    assert out.loc[out["name"] != "Brand New Signing", "has_history"].all()


def test_attach_history_features_uses_current_team_strength_not_blended(
    raw_history: pd.DataFrame, raw_live: pd.DataFrame
) -> None:
    # Transfer one player to a new, much stronger club this season. Their
    # historical row still has the old club's (weaker) team_strength.
    live = raw_live.copy()
    transferred_name = live.loc[0, "name"]
    live.loc[0, "team_id"] = 99
    live.loc[0, "team_strength"] = 2.5
    live.loc[0, "minutes"] = 0  # no minutes yet at the new club

    out = attach_history_features(live, raw_history)
    row = out.loc[out["name"] == transferred_name]
    assert row["team_strength"].iloc[0] == pytest.approx(2.5)


def test_attach_history_features_defaults_available_without_status(
    raw_history: pd.DataFrame, raw_live: pd.DataFrame
) -> None:
    # The fixtures carry no ``status`` column, matching a source that
    # doesn't report it: nothing should be assumed unavailable.
    out = attach_history_features(raw_live, raw_history)
    assert out["available"].all()


def test_attach_history_features_flags_unavailable_status(
    raw_history: pd.DataFrame, raw_live: pd.DataFrame
) -> None:
    live = raw_live.copy()
    live["status"] = "a"
    live.loc[0, "status"] = "i"
    out = attach_history_features(live, raw_history)
    flagged = out.loc[out["name"] == live.loc[0, "name"], "available"]
    assert not flagged.iloc[0]
    assert out["available"].sum() == len(out) - 1
