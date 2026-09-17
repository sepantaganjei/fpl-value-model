"""The price model: a scikit-learn pipeline plus fit/predict/persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fpl_value_model.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_M,
)


def build_pipeline(estimator: RegressorMixin | None = None) -> Pipeline:
    """Assemble the preprocessing + regression pipeline.

    Parameters
    ----------
    estimator
        Regressor for the final step. Defaults to
        :class:`~sklearn.linear_model.LinearRegression`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Standard-scales numeric features, one-hot encodes the position,
        then fits ``estimator``.
    """
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    final = LinearRegression() if estimator is None else estimator
    return Pipeline([("pre", pre), ("model", final)])


def candidate_estimators() -> dict[str, RegressorMixin]:
    """Return the estimators compared during cross-validation."""
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "gbr": GradientBoostingRegressor(random_state=0),
    }


def cross_validate_by_season(frame: pd.DataFrame) -> pd.DataFrame:
    """Score each candidate estimator with season-held-out CV.

    Parameters
    ----------
    frame
        Output of :func:`fpl_value_model.features.build_feature_frame`,
        including a ``season`` column for grouping.

    Returns
    -------
    pandas.DataFrame
        One row per estimator with mean-absolute-error (in millions) and
        R-squared, indexed by estimator name.
    """
    features = frame[list(FEATURE_COLUMNS)]
    target = frame[TARGET_M]
    groups = frame["season"]
    splits = GroupKFold(n_splits=groups.nunique())

    rows = []
    for name, estimator in candidate_estimators().items():
        predicted = cross_val_predict(
            build_pipeline(estimator),
            features,
            target,
            cv=splits,
            groups=groups,
        )
        rows.append(
            {
                "estimator": name,
                "mae_m": mean_absolute_error(target, predicted),
                "r2": r2_score(target, predicted),
            }
        )
    return pd.DataFrame(rows).set_index("estimator").sort_values("mae_m")


@dataclass
class ValueModel:
    """Fitted price model that turns features into a value signal."""

    estimator: RegressorMixin | None = None
    pipeline: Pipeline = field(init=False)

    def __post_init__(self) -> None:
        """Build the underlying pipeline from ``estimator``."""
        self.pipeline = build_pipeline(self.estimator)

    def fit(self, frame: pd.DataFrame) -> ValueModel:
        """Fit on a feature frame carrying the ``price_m`` target.

        Parameters
        ----------
        frame
            Output of
            :func:`fpl_value_model.features.build_feature_frame`.

        Returns
        -------
        ValueModel
            ``self``, fitted.
        """
        self.pipeline.fit(frame[list(FEATURE_COLUMNS)], frame[TARGET_M])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predict fair price in millions for each row of ``frame``.

        Parameters
        ----------
        frame
            Rows containing every name in
            :data:`~fpl_value_model.config.FEATURE_COLUMNS`.

        Returns
        -------
        pandas.Series
            Predicted price in millions, aligned to ``frame.index``.
        """
        values = self.pipeline.predict(frame[list(FEATURE_COLUMNS)])
        return pd.Series(values, index=frame.index, name="pred_m")

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach ``pred_m`` and ``value_m`` (= ``pred_m - price_m``).

        Parameters
        ----------
        frame
            Rows with features and a ``price_m`` column.

        Returns
        -------
        pandas.DataFrame
            Copy of ``frame`` with ``pred_m`` and ``value_m`` columns,
            sorted by ``value_m`` descending (most underpriced first).
        """
        out = frame.copy()
        out["pred_m"] = self.predict(out)
        out["value_m"] = out["pred_m"] - out[TARGET_M]
        return out.sort_values("value_m", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        """Serialise the fitted pipeline to ``path`` with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> ValueModel:
        """Load a model previously written by :meth:`save`."""
        model = cls()
        model.pipeline = joblib.load(path)
        return model
