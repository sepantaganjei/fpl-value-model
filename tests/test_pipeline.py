"""Tests for :mod:`fpl_value_model.pipeline` with data loaders stubbed out."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_value_model import pipeline


@pytest.fixture
def stub_loaders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_history: pd.DataFrame,
    raw_live: pd.DataFrame,
) -> Path:
    monkeypatch.setattr(
        pipeline, "load_training_frame", lambda *, refresh=False: raw_history
    )
    monkeypatch.setattr(
        pipeline, "load_live_players", lambda *, refresh=False: raw_live
    )
    monkeypatch.setattr(pipeline, "PREDICTIONS_PATH", tmp_path / "predictions.parquet")
    monkeypatch.setattr(pipeline, "MODEL_PATH", tmp_path / "model.joblib")
    return tmp_path


def test_run_writes_predictions_and_model(stub_loaders: Path) -> None:
    output = pipeline.run()

    assert (stub_loaders / "predictions.parquet").exists()
    assert (stub_loaders / "model.joblib").exists()

    written = pd.read_parquet(stub_loaders / "predictions.parquet")
    pd.testing.assert_frame_equal(output, written)
    assert {"name", "price_m", "pred_m", "value_m"}.issubset(written.columns)
    assert (written["value_m"] == written["pred_m"] - written["price_m"]).all()
    assert written["value_m"].is_monotonic_decreasing


def test_run_scores_only_players_with_history(stub_loaders: Path) -> None:
    output = pipeline.run()
    # Every live player in the fixture shares a name with history, so all
    # of them should be scored.
    assert len(output) > 0
    assert output["pred_m"].notna().all()
