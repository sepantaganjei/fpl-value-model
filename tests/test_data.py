"""Tests for the schema-normalisation logic in :mod:`fpl_value_model.data`."""

from __future__ import annotations

import pandas as pd

from fpl_value_model.data import _KEEP_COLUMNS, _normalise


def _raw_row() -> dict[str, object]:
    return {
        "id": 42,
        "first_name": "Bukayo",
        "second_name": "Saka",
        "element_type": 3,
        "team": 1,
        "now_cost": 100,
        "total_points": 200,
        "minutes": 3000,
        "goals_scored": 16,
        "assists": 9,
        "expected_goals": "12.5",
        "expected_assists": "8.1",
        "expected_goal_involvements": "20.6",
        "ict_index": "250.4",
        "bonus": 25,
        "clean_sheets": 12,
        "saves": 0,
        "selected_by_percent": "40.1",
        "form": "6.5",
    }


def test_normalise_maps_position_label() -> None:
    out = _normalise(pd.DataFrame([_raw_row()]), season="2024-25")
    assert out.loc[0, "position"] == "MID"


def test_normalise_enforces_shared_schema() -> None:
    out = _normalise(pd.DataFrame([_raw_row()]), season="2024-25")
    assert list(out.columns) == [*_KEEP_COLUMNS, "season"]
    assert out.loc[0, "season"] == "2024-25"


def test_normalise_coerces_numeric_strings() -> None:
    out = _normalise(pd.DataFrame([_raw_row()]), season="2024-25")
    assert out.loc[0, "expected_goals"] == 12.5
    assert pd.api.types.is_numeric_dtype(out["ict_index"])


def test_normalise_fills_missing_columns_with_na() -> None:
    row = _raw_row()
    del row["saves"]
    out = _normalise(pd.DataFrame([row]), season="2024-25")
    assert pd.isna(out.loc[0, "saves"])


def test_normalise_builds_name_when_absent() -> None:
    out = _normalise(pd.DataFrame([_raw_row()]), season="2024-25")
    assert out.loc[0, "name"] == "Bukayo Saka"
