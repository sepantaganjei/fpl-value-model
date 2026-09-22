"""Tests for the schema-normalisation logic in :mod:`fpl_value_model.data`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_value_model.data import _KEEP_COLUMNS, _normalise, _team_strength_table


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


def _raw_teams(overall_home: list[int], overall_away: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, len(overall_home) + 1),
            "strength_overall_home": overall_home,
            "strength_overall_away": overall_away,
        }
    )


def test_team_strength_table_is_zero_mean_unit_std() -> None:
    teams = _raw_teams([1200, 1300, 1400, 1100], [1250, 1350, 1450, 1150])
    out = _team_strength_table(teams)
    assert out["team_strength"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["team_strength"].std() == pytest.approx(1.0)


def test_team_strength_table_ranks_stronger_teams_higher() -> None:
    teams = _raw_teams([1200, 1400], [1200, 1400])
    out = _team_strength_table(teams).set_index("team_id")
    assert out.loc[2, "team_strength"] > out.loc[1, "team_strength"]


def test_team_strength_table_comparable_across_different_absolute_scales() -> None:
    # The live API's compressed 1-5 scale and the historical mirror's
    # ~1200-1400 scale should land in the same normalised range for a
    # team in the same relative position (here, strongest of four).
    historical = _raw_teams([1100, 1200, 1300, 1400], [1100, 1200, 1300, 1400])
    live = _raw_teams([2, 3, 4, 5], [2, 3, 4, 5])
    hist_top = _team_strength_table(historical)["team_strength"].iloc[-1]
    live_top = _team_strength_table(live)["team_strength"].iloc[-1]
    assert hist_top == pytest.approx(live_top)


def test_team_strength_table_handles_zero_variance() -> None:
    # All teams rated identically (e.g. the live API's zeroed sub-fields)
    # must not produce inf/NaN from a zero standard deviation.
    teams = _raw_teams([0, 0, 0], [0, 0, 0])
    out = _team_strength_table(teams)
    assert np.isfinite(out["team_strength"]).all()
    assert (out["team_strength"] == 0.0).all()
