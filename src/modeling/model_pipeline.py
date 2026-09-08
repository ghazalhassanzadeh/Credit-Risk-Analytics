"""Build model pipelines and calculate aggregate evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)

RANDOM_SEED = 42
MIN_CATEGORY_COUNT = 500
OTHER_CATEGORY = "Other/Unknown"
REVIEW_CAPACITIES = (0.05, 0.10, 0.20)

CATEGORICAL_FEATURES = {
    "naics2",
    "project_state",
    "business_age",
    "business_type",
    "processing_method",
}

DEFAULT_BOOSTING_PARAMETERS = {
    "learning_rate": 0.08,
    "max_iter": 150,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 500,
    "l2_regularization": 1.0,
}


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace infrequent and unseen categories using training counts only."""

    def __init__(self, min_count: int = MIN_CATEGORY_COUNT):
        self.min_count = min_count

    def fit(self, X, y=None):
        """Record categories meeting the minimum frequency in the training data."""
        frame = pd.DataFrame(X).astype(str)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.kept_categories_ = {
            column: set(
                frame[column]
                .value_counts()
                .loc[lambda counts: counts >= self.min_count]
                .index
            )
            for column in frame.columns
        }
        return self

    def transform(self, X):
        """Map categories below the training threshold to ``Other/Unknown``."""
        frame = pd.DataFrame(X, columns=self.feature_names_in_).astype(str).copy()
        for column in frame.columns:
            frame[column] = frame[column].where(
                frame[column].isin(self.kept_categories_[column]),
                OTHER_CATEGORY,
            )
        return frame

    def get_feature_names_out(self, input_features=None):
        """Return the input names because grouping does not add columns."""
        return self.feature_names_in_


def _log1p(values):
    """Apply the approved log transformation to gross approval values."""
    return np.log1p(np.asarray(values, dtype=float))


def _split_feature_types(features: Sequence[str]) -> tuple[list[str], list[str]]:
    continuous = [name for name in features if name not in CATEGORICAL_FEATURES]
    categorical = [name for name in features if name in CATEGORICAL_FEATURES]
    return continuous, categorical


def logistic(features: Sequence[str], form: str = "simple", C: float = 0.1):
    """Create the regularized logistic-regression baseline pipeline."""
    if form not in {"simple", "spline"}:
        raise ValueError("form must be 'simple' or 'spline'")

    continuous_features, categorical_features = _split_feature_types(features)
    transformers = []

    for feature in continuous_features:
        steps = []
        if feature == "gross_approval":
            steps.append(
                (
                    "log1p",
                    FunctionTransformer(_log1p, feature_names_out="one-to-one"),
                )
            )
        if form == "spline":
            steps.append(
                (
                    "spline",
                    SplineTransformer(
                        n_knots=4,
                        degree=2,
                        include_bias=False,
                        knots="quantile",
                    ),
                )
            )
        steps.append(("scale", StandardScaler()))
        transformers.append((feature, Pipeline(steps), [feature]))

    categorical_pipeline = Pipeline(
        [
            ("rare", RareCategoryGrouper(MIN_CATEGORY_COUNT)),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    transformers.append(("cat", categorical_pipeline, categorical_features))

    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transformers, sparse_threshold=0.3),
            ),
            (
                "model",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=500,
                    tol=1e-6,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def boosting(features: Sequence[str], params: dict | None = None):
    """Create the histogram gradient-boosting pipeline."""
    model_parameters = (
        DEFAULT_BOOSTING_PARAMETERS.copy() if params is None else params.copy()
    )
    continuous_features, categorical_features = _split_feature_types(features)

    categorical_pipeline = Pipeline(
        [
            ("rare", RareCategoryGrouper(MIN_CATEGORY_COUNT)),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                ),
            ),
        ]
    )
    transformers = [
        ("continuous", "passthrough", continuous_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
    category_start = len(continuous_features)
    category_stop = category_start + len(categorical_features)

    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transformers, sparse_threshold=0),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    categorical_features=list(range(category_start, category_stop)),
                    early_stopping=False,
                    random_state=RANDOM_SEED,
                    **model_parameters,
                ),
            ),
        ]
    )


def cal_params(outcomes, probabilities) -> tuple[float, float]:
    """Estimate calibration intercept and slope from predicted probabilities."""
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=200,
    ).fit(log_odds, outcomes)
    intercept = float(calibration_model.intercept_[0])
    slope = float(calibration_model.coef_[0, 0])
    return intercept, slope


def metrics(outcomes, probabilities) -> dict[str, float | int]:
    """Calculate aggregate ranking, probability, and calibration measures."""
    intercept, slope = cal_params(outcomes, probabilities)
    return {
        "loans": len(outcomes),
        "chargeoffs": int(np.sum(outcomes)),
        "prevalence": float(np.mean(outcomes)),
        "pr_auc": average_precision_score(outcomes, probabilities),
        "roc_auc": roc_auc_score(outcomes, probabilities),
        "brier": brier_score_loss(outcomes, probabilities),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_predicted_risk": float(np.mean(probabilities)),
    }


def capacities(outcomes, probabilities) -> list[dict[str, float | int]]:
    """Summarize retrospective results at the approved review capacities."""
    ranking = np.argsort(-probabilities, kind="stable")
    total_chargeoffs = np.sum(outcomes)
    prevalence = np.mean(outcomes)
    results = []

    for capacity in REVIEW_CAPACITIES:
        reviewed_count = int(np.ceil(capacity * len(outcomes)))
        captured_count = int(np.sum(outcomes[ranking[:reviewed_count]]))
        precision = captured_count / reviewed_count
        capture = captured_count / total_chargeoffs
        results.append(
            {
                "capacity_percent": 100 * capacity,
                "reviewed_loans": reviewed_count,
                "captured_chargeoffs": captured_count,
                "capture_percent": 100 * capture,
                "precision_percent": 100 * precision,
                "lift": precision / prevalence,
            }
        )

    return results


def cal_bins(outcomes, probabilities) -> pd.DataFrame:
    """Create decile-level observed and predicted risks for calibration plots."""
    calibration = pd.DataFrame(
        {"outcome": outcomes, "predicted_probability": probabilities}
    )
    calibration["bin"] = pd.qcut(
        calibration["predicted_probability"],
        10,
        duplicates="drop",
    )
    result = (
        calibration.groupby("bin", observed=True)
        .agg(
            loans=("outcome", "size"),
            chargeoffs=("outcome", "sum"),
            mean_predicted_risk=("predicted_probability", "mean"),
            observed_risk=("outcome", "mean"),
        )
        .reset_index(drop=True)
    )
    result.insert(0, "risk_bin", range(1, len(result) + 1))
    return result


def unseen(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: Sequence[str],
) -> list[dict[str, float | int | str]]:
    """Report rare and unseen category rates without exposing loan-level data."""
    results = []

    for feature in [name for name in features if name in CATEGORICAL_FEATURES]:
        training_counts = training_data[feature].astype(str).value_counts()
        kept_categories = set(
            training_counts.loc[
                lambda counts: counts >= MIN_CATEGORY_COUNT
            ].index
        )
        known_categories = set(training_counts.index)
        test_values = test_data[feature].astype(str)

        results.append(
            {
                "feature": feature,
                "other_or_unknown_percent": 100
                * (~test_values.isin(kept_categories)).mean(),
                "strictly_unseen_percent": 100
                * (~test_values.isin(known_categories)).mean(),
                "training_kept_categories": len(kept_categories),
            }
        )

    return results
