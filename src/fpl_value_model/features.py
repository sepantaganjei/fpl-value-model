"""Turn raw player-season rows into the model's feature matrix.

Counting stats (goals, xG, ICT, ...) are converted to per-90 rates so a
part-season and a full season are comparable, and player-seasons below a
minutes threshold are dropped because their rates are too noisy to price
against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_value_model.config import (
    COUNTING_STATS,
    FEATURE_COLUMNS,
    MIN_MINUTES,
    TARGET_COLUMN,
    TARGET_M,
)


def add_price_m(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``price_m`` (FPL price in millions) from ``now_cost``.

    Parameters
    ----------
    frame
        Rows containing the raw :data:`~fpl_value_model.config.TARGET_COLUMN`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with a ``price_m`` column.
    """
    out = frame.copy()
    out[TARGET_M] = out[TARGET_COLUMN] / 10.0
    return out


def to_per90(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``<stat>_per90`` columns for every counting stat.

    Rows with zero minutes yield zero rates rather than division errors.

    Parameters
    ----------
    frame
        Rows containing ``minutes`` and every name in
        :data:`~fpl_value_model.config.COUNTING_STATS`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with the added per-90 columns.
    """
    out = frame.copy()
    minutes = out["minutes"].to_numpy(dtype="float64")
    safe = np.where(minutes > 0, minutes, np.nan)
    for stat in COUNTING_STATS:
        rate = out[stat].to_numpy(dtype="float64") / safe * 90.0
        out[f"{stat}_per90"] = np.nan_to_num(rate, nan=0.0)
    return out


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    min_minutes: int = MIN_MINUTES,
    require_target: bool = True,
) -> pd.DataFrame:
    """Produce a clean, model-ready frame from raw player-season rows.

    Steps: add ``price_m``, add per-90 rates, drop player-seasons below
    ``min_minutes``, and drop rows with missing features (or target).

    Parameters
    ----------
    frame
        Raw player-season rows on the shared schema from
        :mod:`fpl_value_model.data`.
    min_minutes
        Minimum season minutes to keep a row.
    require_target
        When ``True``, also drop rows with no ``price_m``.

    Returns
    -------
    pandas.DataFrame
        Rows carrying identifier columns, ``price_m``, and every name in
        :data:`~fpl_value_model.config.FEATURE_COLUMNS`.
    """
    out = add_price_m(frame)
    out = to_per90(out)
    out = out[out["minutes"].fillna(0) >= min_minutes]

    required = list(FEATURE_COLUMNS)
    if require_target:
        required = [*required, TARGET_M]
    out = out.dropna(subset=required)

    keep = [
        c
        for c in ("player_id", "name", "season", "position", "team_id", TARGET_M)
        if c in out.columns
    ]
    keep += [c for c in FEATURE_COLUMNS if c not in keep]
    return out[keep].reset_index(drop=True)


def latest_history_per_player(history: pd.DataFrame) -> pd.DataFrame:
    """Reduce historical player-seasons to each player's most recent one.

    Parameters
    ----------
    history
        Output of :func:`build_feature_frame` on historical data, with a
        ``season`` column such as ``"2024-25"``.

    Returns
    -------
    pandas.DataFrame
        One row per player name, from their latest season.
    """
    ordered = history.sort_values("season")
    return ordered.drop_duplicates("name", keep="last").reset_index(drop=True)


def attach_history_features(live: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Give each live player the per-90 features from their latest season.

    Live players with no historical match (new signings, academy players)
    are dropped: the model has nothing to price them on.

    Parameters
    ----------
    live
        Current-season players on the shared schema.
    history
        Raw historical player-season rows (pre-feature-build).

    Returns
    -------
    pandas.DataFrame
        Live players with current ``price_m`` and historical features,
        ready for :meth:`fpl_value_model.model.ValueModel.predict`.
    """
    hist_features = latest_history_per_player(
        build_feature_frame(history, require_target=False)
    )
    feature_cols = list(FEATURE_COLUMNS)

    id_cols = [
        c
        for c in ("player_id", "name", "position", "team_id", "total_points")
        if c in live.columns
    ]
    live_priced = add_price_m(live)[[*id_cols, TARGET_M]]
    merged = live_priced.merge(
        hist_features[["name", *feature_cols]],
        on="name",
        how="inner",
        suffixes=("", "_hist"),
    )
    # Prefer the live position label over the historical one.
    if "position_hist" in merged.columns:
        merged = merged.drop(columns="position_hist")
    return merged.reset_index(drop=True)
