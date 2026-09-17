"""Walk-forward backtest of the value signal against simple baselines.

For each season after the first, the model is fitted on all earlier
seasons and used to rank the previous season's players. The top ``N`` of
each ranking are then judged on what they actually returned the following
season, measured as points per million of acquisition cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_value_model.config import MIN_MINUTES, TARGET_M
from fpl_value_model.features import add_price_m, build_feature_frame
from fpl_value_model.model import ValueModel


def _outcome_table(history_raw: pd.DataFrame) -> pd.DataFrame:
    """Per player-season points, form, price, and season points-per-90."""
    table = add_price_m(history_raw)[
        ["season", "name", "total_points", "form", "minutes", TARGET_M]
    ].copy()
    table["points_per90"] = np.where(
        table["minutes"] > 0,
        table["total_points"] / table["minutes"] * 90.0,
        0.0,
    )
    return table


#: Ranking key -> (column, ascending) for each strategy under test.
_STRATEGIES: dict[str, tuple[str, bool]] = {
    "model_value": ("value_m", False),
    "fpl_form": ("form", False),
    "cheapest": (TARGET_M, True),
    "points_per90": ("points_per90", False),
}


def walk_forward_backtest(
    history_raw: pd.DataFrame,
    *,
    top_n: int = 20,
    min_minutes: int = MIN_MINUTES,
) -> pd.DataFrame:
    """Run the walk-forward comparison.

    Parameters
    ----------
    history_raw
        Raw historical player-season rows on the shared schema (the
        output of :func:`fpl_value_model.data.load_training_frame`).
    top_n
        Squad size drafted by each strategy per season.
    min_minutes
        Minutes threshold for a player-season to be a candidate.

    Returns
    -------
    pandas.DataFrame
        Indexed by strategy, one column per test season plus ``mean``,
        holding the mean forward points-per-million of the drafted top
        ``top_n``. Higher is better.
    """
    features = build_feature_frame(
        history_raw, min_minutes=min_minutes, require_target=True
    )
    outcomes = _outcome_table(history_raw)
    seasons = sorted(features["season"].unique())
    if len(seasons) < 2:
        raise ValueError("need at least two seasons to backtest")

    results: dict[str, dict[str, float]] = {name: {} for name in _STRATEGIES}

    for test_season in seasons[1:]:
        prior_season = seasons[seasons.index(test_season) - 1]
        train = features[features["season"] < test_season]
        candidates = features[features["season"] == prior_season]
        if train.empty or candidates.empty:
            continue

        scored = ValueModel().fit(train).score_frame(candidates)
        scored = scored.merge(
            outcomes[outcomes["season"] == prior_season].drop(columns=[TARGET_M]),
            on=["season", "name"],
            how="left",
        )
        forward = outcomes.loc[
            outcomes["season"] == test_season, ["name", "total_points"]
        ].rename(columns={"total_points": "forward_points"})
        scored = scored.merge(forward, on="name", how="inner")
        scored = scored[scored[TARGET_M] > 0]
        scored["forward_ppm"] = scored["forward_points"] / scored[TARGET_M]

        for name, (column, ascending) in _STRATEGIES.items():
            drafted = scored.sort_values(column, ascending=ascending).head(top_n)
            results[name][test_season] = float(drafted["forward_ppm"].mean())

    frame = pd.DataFrame(results).T
    frame["mean"] = frame.mean(axis=1)
    return frame.sort_values("mean", ascending=False)
