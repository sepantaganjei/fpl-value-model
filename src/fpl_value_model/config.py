"""Project-wide paths, constants, and column definitions."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

#: Committed artefact the Streamlit app reads; refreshed by the daily job.
PREDICTIONS_PATH: Path = DATA_DIR / "predictions.parquet"
#: Serialised fitted model.
MODEL_PATH: Path = DATA_DIR / "model.joblib"

FPL_BOOTSTRAP_URL: str = "https://fantasy.premierleague.com/api/bootstrap-static/"

VAASTAV_PLAYERS_RAW_URL: str = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/players_raw.csv"
)

#: Historical seasons that expose the ``expected_*`` columns.
HISTORICAL_SEASONS: tuple[str, ...] = ("2022-23", "2023-24", "2024-25", "2025-26")

#: FPL ``element_type`` code -> position label.
POSITION_MAP: dict[int, str] = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

#: Raw counting stats pulled from source, converted to per-90 rates.
COUNTING_STATS: tuple[str, ...] = (
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
    "bonus",
    "clean_sheets",
    "saves",
)

#: Per-90 feature columns produced by :func:`fpl_value_model.features.to_per90`.
PER90_FEATURES: tuple[str, ...] = tuple(f"{stat}_per90" for stat in COUNTING_STATS)

#: Numeric model features: per-90 rates plus season minutes as a role proxy.
NUMERIC_FEATURES: tuple[str, ...] = (*PER90_FEATURES, "minutes")

#: Categorical model features, one-hot encoded before fitting.
CATEGORICAL_FEATURES: tuple[str, ...] = ("position",)

#: All model input columns.
FEATURE_COLUMNS: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

#: Raw regression target: FPL price in tenths of a million.
TARGET_COLUMN: str = "now_cost"
#: Modelling target: FPL price in millions (``now_cost / 10``).
TARGET_M: str = "price_m"

#: Minimum season minutes for a player-season to carry a stable per-90 signal.
MIN_MINUTES: int = 450

#: FPL ``status`` codes meaning the player can actually be picked right now.
AVAILABLE_STATUSES: frozenset[str] = frozenset({"a"})


def ensure_data_dirs() -> None:
    """Create the ``data/`` subdirectories if they do not yet exist."""
    for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)
