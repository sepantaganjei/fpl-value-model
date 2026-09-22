"""End-to-end run: fit on history, score the live pool, write predictions.

Invoke with ``python -m fpl_value_model.pipeline`` (or ``make train``).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from fpl_value_model.config import (
    MODEL_PATH,
    PREDICTIONS_PATH,
    TARGET_M,
    ensure_data_dirs,
)
from fpl_value_model.data import load_live_players, load_training_frame
from fpl_value_model.features import attach_history_features, build_feature_frame
from fpl_value_model.model import ValueModel, cross_validate_by_season

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS: tuple[str, ...] = (
    "player_id",
    "name",
    "position",
    "team_id",
    "price_m",
    "pred_m",
    "value_m",
    "total_points",
    "points_per_m",
    "available",
    "status",
    "chance_of_playing_next_round",
    "has_history",
)


def run(*, refresh: bool = False) -> pd.DataFrame:
    """Fit the model and write live predictions to ``PREDICTIONS_PATH``.

    Parameters
    ----------
    refresh
        Re-download source data instead of using the parquet caches.

    Returns
    -------
    pandas.DataFrame
        The written predictions, sorted most-underpriced first.
    """
    ensure_data_dirs()

    history = load_training_frame(refresh=refresh)
    features = build_feature_frame(history)
    logger.info("training rows: %d", len(features))

    cv = cross_validate_by_season(features)
    logger.info("season-held-out CV:\n%s", cv.round(3).to_string())

    model = ValueModel().fit(features)
    model.save(MODEL_PATH)

    live = load_live_players(refresh=refresh)
    scored = model.score_frame(attach_history_features(live, history))

    scored["total_points"] = scored.get("total_points", pd.Series(dtype="float64"))
    scored["points_per_m"] = scored["total_points"] / scored[TARGET_M]
    output = scored[[c for c in _OUTPUT_COLUMNS if c in scored.columns]]

    output.to_parquet(PREDICTIONS_PATH, index=False)
    logger.info("wrote %d predictions to %s", len(output), PREDICTIONS_PATH)
    return output


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download source data before running",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(refresh=args.refresh)


if __name__ == "__main__":
    main()
