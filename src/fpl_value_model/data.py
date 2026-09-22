"""Fetch and cache Fantasy Premier League player data.

Two sources are used:

* the live ``bootstrap-static`` endpoint for the current season, and
* the ``vaastav/Fantasy-Premier-League`` GitHub mirror for past seasons.

Both are free and require no API key.
"""

from __future__ import annotations

import argparse
import io
from typing import Any, cast

import pandas as pd
import requests

from fpl_value_model.config import (
    FPL_BOOTSTRAP_URL,
    HISTORICAL_SEASONS,
    POSITION_MAP,
    RAW_DIR,
    VAASTAV_PLAYERS_RAW_URL,
    VAASTAV_TEAMS_RAW_URL,
    ensure_data_dirs,
)

_REQUEST_TIMEOUT: int = 30
_USER_AGENT: str = "fpl-value-model/0.1 (+https://github.com/sepantaganjei)"

#: Columns kept from every raw source, aligned to a single schema.
_KEEP_COLUMNS: tuple[str, ...] = (
    "player_id",
    "name",
    "position",
    "team_id",
    "now_cost",
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
    "bonus",
    "clean_sheets",
    "saves",
    "selected_by_percent",
    "form",
    "status",
    "chance_of_playing_next_round",
)


def _get(url: str) -> requests.Response:
    """Issue a GET request with a project user-agent and shared timeout.

    Parameters
    ----------
    url
        Absolute URL to fetch.

    Returns
    -------
    requests.Response
        The successful response (``raise_for_status`` has been called).
    """
    response = requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response


def fetch_bootstrap() -> dict[str, Any]:
    """Download the raw ``bootstrap-static`` payload for the live season.

    Returns
    -------
    dict
        The decoded JSON response, whose ``"elements"`` key holds one
        record per player.
    """
    return cast("dict[str, Any]", _get(FPL_BOOTSTRAP_URL).json())


def _normalise(frame: pd.DataFrame, *, season: str) -> pd.DataFrame:
    """Coerce a raw player frame onto the shared schema.

    Parameters
    ----------
    frame
        Raw player rows with FPL's native column names.
    season
        Season label (e.g. ``"2024-25"``) added as a column.

    Returns
    -------
    pandas.DataFrame
        Frame restricted to :data:`_KEEP_COLUMNS` plus ``season``, with
        numeric columns cast to floats and ``position`` mapped to labels.
    """
    out = frame.rename(
        columns={
            "id": "player_id",
            "element_type": "position",
            "team": "team_id",
        }
    )
    if "name" not in out.columns:
        out["name"] = (
            out["first_name"].fillna("") + " " + out["second_name"].fillna("")
        ).str.strip()

    out["position"] = out["position"].map(POSITION_MAP)

    for column in _KEEP_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    out = out[list(_KEEP_COLUMNS)].copy()
    out["season"] = season

    numeric = [c for c in _KEEP_COLUMNS if c not in {"name", "position", "status"}]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    return out


def _team_strength_table(teams: pd.DataFrame) -> pd.DataFrame:
    """Reduce a raw FPL teams table to a within-source z-scored rating.

    FPL's absolute strength scale has drifted over time and even differs
    between the live API and the historical mirror (a ~1-5 scale live vs.
    a ~1200-1400 scale historically), so a raw value isn't comparable
    across seasons. Z-scoring within the season (or the live pool) turns
    it into "how strong is this team relative to its 19 rivals that
    season", which is comparable everywhere and is what actually matters
    for pricing a player.

    Parameters
    ----------
    teams
        A raw ``teams`` table (FPL's ``bootstrap-static`` teams array, or
        a season's vaastav ``teams.csv``), with ``id``,
        ``strength_overall_home``, and ``strength_overall_away`` columns.

    Returns
    -------
    pandas.DataFrame
        One row per team: ``team_id`` and ``team_strength``.
    """
    home = pd.to_numeric(teams["strength_overall_home"], errors="coerce")
    away = pd.to_numeric(teams["strength_overall_away"], errors="coerce")
    overall = (home + away) / 2.0
    std = overall.std()
    z = (overall - overall.mean()) / std if std and not pd.isna(std) else overall * 0.0
    return pd.DataFrame(
        {
            "team_id": pd.to_numeric(teams["id"], errors="coerce"),
            "team_strength": z,
        }
    )


def _team_names_table(teams: pd.DataFrame) -> pd.DataFrame:
    """Reduce a raw FPL teams table to an id -> display-name lookup."""
    return pd.DataFrame(
        {
            "team_id": pd.to_numeric(teams["id"], errors="coerce"),
            "team_name": teams["name"],
        }
    )


def _fetch_teams_csv(season: str) -> pd.DataFrame:
    """Download and parse a season's ``teams.csv`` from the vaastav mirror."""
    response = _get(VAASTAV_TEAMS_RAW_URL.format(season=season))
    response.encoding = "utf-8"
    return pd.read_csv(io.StringIO(response.text))


def load_live_players(*, refresh: bool = False) -> pd.DataFrame:
    """Return the current-season player table, cached to ``data/raw``.

    Parameters
    ----------
    refresh
        When ``True``, re-download even if a cached parquet exists.

    Returns
    -------
    pandas.DataFrame
        One row per player on the shared schema, ``season == "live"``.
    """
    ensure_data_dirs()
    cache = RAW_DIR / "players_live.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    payload = fetch_bootstrap()
    elements = pd.DataFrame(payload["elements"])
    frame = _normalise(elements, season="live")

    teams = pd.DataFrame(payload["teams"])
    frame = frame.merge(_team_strength_table(teams), on="team_id", how="left")
    frame = frame.merge(_team_names_table(teams), on="team_id", how="left")

    frame.to_parquet(cache, index=False)
    return frame


def load_historical_players(
    seasons: tuple[str, ...] = HISTORICAL_SEASONS,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return concatenated player tables for past seasons.

    Parameters
    ----------
    seasons
        Season labels to download from the vaastav mirror.
    refresh
        When ``True``, re-download even if a cached parquet exists.

    Returns
    -------
    pandas.DataFrame
        One row per player-season on the shared schema.
    """
    ensure_data_dirs()
    cache = RAW_DIR / "players_history.parquet"
    if cache.exists() and not refresh:
        cached = pd.read_parquet(cache)
        if set(seasons).issubset(set(cached["season"].unique())):
            return cached[cached["season"].isin(seasons)].reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for season in seasons:
        url = VAASTAV_PLAYERS_RAW_URL.format(season=season)
        response = _get(url)
        response.encoding = "utf-8"
        raw = pd.read_csv(io.StringIO(response.text))
        frame = _normalise(raw, season=season)

        teams = _fetch_teams_csv(season)
        frame = frame.merge(_team_strength_table(teams), on="team_id", how="left")
        frame = frame.merge(_team_names_table(teams), on="team_id", how="left")

        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(cache, index=False)
    return combined


def load_training_frame(*, refresh: bool = False) -> pd.DataFrame:
    """Return historical player-seasons ready for model fitting.

    Rows with no recorded minutes are dropped, since a player who never
    played carries no performance signal to price against.

    Parameters
    ----------
    refresh
        Forwarded to the underlying loaders.

    Returns
    -------
    pandas.DataFrame
        Cleaned historical player-seasons.
    """
    frame = load_historical_players(refresh=refresh)
    frame = frame[frame["minutes"].fillna(0) > 0].reset_index(drop=True)
    return frame


def _main() -> None:
    """Refresh both caches from the CLI (``python -m fpl_value_model.data``)."""
    parser = argparse.ArgumentParser(description="Refresh cached FPL data.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download even if a cache exists",
    )
    args = parser.parse_args()

    live = load_live_players(refresh=args.refresh)
    history = load_historical_players(refresh=args.refresh)
    print(f"live players:      {len(live):>5}")
    print(f"historical rows:   {len(history):>5}")
    print(f"cache directory:   {RAW_DIR}")


if __name__ == "__main__":
    _main()
