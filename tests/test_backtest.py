"""Tests for :mod:`fpl_value_model.backtest`."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_value_model.backtest import walk_forward_backtest


def test_backtest_shape_and_strategies(raw_history: pd.DataFrame) -> None:
    result = walk_forward_backtest(raw_history, top_n=10, min_minutes=300)
    assert "mean" in result.columns
    assert {"model_value", "fpl_form", "cheapest", "points_per90"}.issubset(
        result.index
    )
    # One column per test season (seasons after the first) plus ``mean``.
    assert len(result.columns) >= 2
    assert result["mean"].notna().all()


def test_backtest_is_sorted_by_mean(raw_history: pd.DataFrame) -> None:
    result = walk_forward_backtest(raw_history, top_n=10, min_minutes=300)
    assert result["mean"].is_monotonic_decreasing


def test_backtest_needs_two_seasons(raw_history: pd.DataFrame) -> None:
    one_season = raw_history[raw_history["season"] == "2022-23"]
    with pytest.raises(ValueError, match="two seasons"):
        walk_forward_backtest(one_season)
