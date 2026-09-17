"""Tests for :mod:`fpl_value_model.model`."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

from fpl_value_model.features import attach_history_features, build_feature_frame
from fpl_value_model.model import (
    ValueModel,
    build_pipeline,
    cross_validate_by_season,
)


def test_build_pipeline_has_preprocess_and_model() -> None:
    pipe = build_pipeline()
    assert list(pipe.named_steps) == ["pre", "model"]
    assert isinstance(pipe.named_steps["model"], LinearRegression)


def test_value_model_fit_predict_shapes(raw_history: pd.DataFrame) -> None:
    frame = build_feature_frame(raw_history)
    model = ValueModel().fit(frame)
    preds = model.predict(frame)
    assert len(preds) == len(frame)
    assert preds.notna().all()


def test_score_frame_adds_value_and_sorts(raw_history: pd.DataFrame) -> None:
    frame = build_feature_frame(raw_history)
    scored = ValueModel().fit(frame).score_frame(frame)
    assert {"pred_m", "value_m"}.issubset(scored.columns)
    assert (scored["value_m"] == scored["pred_m"] - scored["price_m"]).all()
    assert scored["value_m"].is_monotonic_decreasing


def test_cross_validate_by_season_ranks_models(raw_history: pd.DataFrame) -> None:
    frame = build_feature_frame(raw_history)
    table = cross_validate_by_season(frame)
    assert {"mae_m", "r2"}.issubset(table.columns)
    assert "linear" in table.index
    # A real estimator should beat predicting the mean.
    assert table.loc["linear", "mae_m"] <= table.loc["baseline_mean", "mae_m"]


def test_save_load_round_trip(
    raw_history: pd.DataFrame, raw_live: pd.DataFrame, tmp_path: object
) -> None:
    frame = build_feature_frame(raw_history)
    model = ValueModel().fit(frame)
    live = attach_history_features(raw_live, raw_history)
    before = model.predict(live)

    path = tmp_path / "model.joblib"  # type: ignore[operator]
    model.save(path)
    after = ValueModel.load(path).predict(live)
    pd.testing.assert_series_equal(before, after)
