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
    AVAILABLE_STATUSES,
    BLEND_FEATURES,
    COUNTING_STATS,
    CURRENT_FEATURES,
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


def attach_history_features(
    live: pd.DataFrame,
    history: pd.DataFrame,
    *,
    shrinkage: int = MIN_MINUTES,
) -> pd.DataFrame:
    """Blend each live player's in-progress season with their latest one.

    This season's per-90 rates and minutes are shrunk toward a prior,
    weighted by how many minutes they've played so far this season: early
    on, with only a handful of minutes, the blend leans on the prior since
    a tiny in-season sample is too noisy to trust; as minutes accumulate
    this season it shifts toward the player's current form. The prior is
    the player's own most recent completed season where there is one; for
    a live player with no historical match (new signings, promoted-team
    players, academy graduates) it falls back to the average of their
    position instead of being dropped, since even a rough position-shaped
    prior beats no prediction at all. Current-state features (see
    :data:`~fpl_value_model.config.CURRENT_FEATURES`) are the exception:
    they come straight from the player's current club and current
    ownership, unblended, since a summer transfer should see their new
    team's strength immediately rather than a mix with their old one's,
    and this week's ownership is what the market thinks right now, not a
    form stat to smooth from a small in-season sample.

    Parameters
    ----------
    live
        Current-season players on the shared schema.
    history
        Raw historical player-season rows (pre-feature-build).
    shrinkage
        Minutes played this season at which the blend is 50/50 between
        this season and last. Defaults to :data:`MIN_MINUTES`, the same
        threshold used elsewhere to call a per-90 rate stable.

    Returns
    -------
    pandas.DataFrame
        Every live player with current ``price_m`` and blended features,
        ready for :meth:`fpl_value_model.model.ValueModel.predict`. Carries
        a ``has_history`` flag: ``False`` means the prediction rests on a
        position average rather than the player's own record, and should
        be trusted less.
    """
    hist_features = latest_history_per_player(
        build_feature_frame(history, require_target=False)
    )
    numeric_cols = list(BLEND_FEATURES)
    position_avg = hist_features.groupby("position")[numeric_cols].mean()

    id_cols = [
        c
        for c in (
            "player_id",
            "name",
            "position",
            "team_id",
            "total_points",
            "status",
            "chance_of_playing_next_round",
            *CURRENT_FEATURES,
        )
        if c in live.columns
    ]
    live_current = to_per90(live)
    live_priced = add_price_m(live_current)[[*id_cols, TARGET_M, *numeric_cols]]
    merged = live_priced.merge(
        hist_features[["name", *numeric_cols]],
        on="name",
        how="left",
        suffixes=("_now", "_hist"),
    )
    had_history = merged["minutes_hist"].notna()
    for col in numeric_cols:
        fallback = merged["position"].map(position_avg[col])
        merged[f"{col}_hist"] = merged[f"{col}_hist"].fillna(fallback)

    weight = merged["minutes_now"] / (merged["minutes_now"] + shrinkage)
    for col in numeric_cols:
        merged[col] = (
            weight * merged[f"{col}_now"] + (1 - weight) * merged[f"{col}_hist"]
        )
        merged = merged.drop(columns=[f"{col}_now", f"{col}_hist"])

    if "status" in merged.columns:
        merged["available"] = merged["status"].isin(AVAILABLE_STATUSES)
    else:
        merged["available"] = True
    merged["has_history"] = had_history.to_numpy()

    return merged.reset_index(drop=True)
