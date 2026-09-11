"""Small, optional helpers for pion-vs-shower classifier training.

Scikit-learn is imported only when training is requested, so feature extraction
does not acquire a new mandatory dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings

import numpy as np


@dataclass
class ClassifierTrainingResult:
    estimator: Any
    feature_names: list[str]
    out_of_fold_pion_probability: np.ndarray
    fold_index: np.ndarray
    metrics: dict[str, float]


def threshold_for_target_pion_efficiency(
    labels: Sequence[int],
    pion_probabilities: Sequence[float],
    target_efficiency: float = 0.80,
) -> float:
    """Choose the highest practical score cut retaining a target pion fraction."""

    y = np.asarray(labels, dtype=int)
    score = np.asarray(pion_probabilities, dtype=np.float64)
    if y.shape != score.shape or y.ndim != 1:
        raise ValueError("labels and pion_probabilities must be same-length 1D arrays.")
    if not 0.0 < target_efficiency <= 1.0:
        raise ValueError("target_efficiency must lie in (0, 1].")
    pion_scores = np.sort(score[(y == 1) & np.isfinite(score)])
    if pion_scores.size == 0:
        raise ValueError("No finite pion scores were supplied.")
    retained = int(np.ceil(target_efficiency * pion_scores.size))
    index = max(0, pion_scores.size - retained)
    return float(pion_scores[index])


def classification_metrics_at_threshold(
    labels: Sequence[int],
    pion_probabilities: Sequence[float],
    threshold: float,
) -> dict[str, float]:
    """Return pion efficiency/purity and shower rejection at a score cut."""

    y = np.asarray(labels, dtype=int)
    score = np.asarray(pion_probabilities, dtype=np.float64)
    if y.shape != score.shape or y.ndim != 1:
        raise ValueError("labels and pion_probabilities must be same-length 1D arrays.")
    prediction = score >= float(threshold)
    pion = y == 1
    shower = y == 0
    true_positive = int(np.count_nonzero(prediction & pion))
    false_positive = int(np.count_nonzero(prediction & shower))
    n_pion = int(np.count_nonzero(pion))
    n_shower = int(np.count_nonzero(shower))
    selected = true_positive + false_positive
    return {
        "threshold": float(threshold),
        "pion_efficiency": true_positive / n_pion if n_pion else float("nan"),
        "pion_purity": true_positive / selected if selected else float("nan"),
        "shower_misidentification": false_positive / n_shower if n_shower else float("nan"),
        "shower_rejection": 1.0 - false_positive / n_shower if n_shower else float("nan"),
    }


def _feature_matrix(
    feature_rows: Sequence[Mapping[str, float]],
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if not feature_rows:
        raise ValueError("feature_rows is empty.")
    if feature_names is None:
        common = set(feature_rows[0])
        for row in feature_rows[1:]:
            common.intersection_update(row)
        names = sorted(common)
    else:
        names = list(feature_names)
    if not names:
        raise ValueError("No classifier features were selected.")
    matrix = np.asarray(
        [[float(row.get(name, np.nan)) for name in names] for row in feature_rows],
        dtype=np.float64,
    )
    return matrix, names


def train_event_classifier(
    feature_rows: Sequence[Mapping[str, float]],
    labels: Sequence[int],
    *,
    groups: Sequence[Any] | None = None,
    feature_names: Sequence[str] | None = None,
    model: str = "logistic",
    n_splits: int = 5,
    random_state: int = 12345,
) -> ClassifierTrainingResult:
    """Train a baseline classifier and return leakage-resistant OOF predictions.

    Use ``groups`` for simulation production, run, or file identifiers so that
    closely related events cannot appear in both train and validation folds.
    """

    try:
        from sklearn.base import clone
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "Classifier training is optional and requires scikit-learn. "
            "Install it with `python -m pip install scikit-learn`."
        ) from exc

    matrix, names = _feature_matrix(feature_rows, feature_names)
    y = np.asarray(labels, dtype=int)
    if y.shape != (matrix.shape[0],) or not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must contain one binary (0=shower, 1=pion) value per row.")

    model_key = str(model).strip().lower()
    if model_key in {"logistic", "logistic_regression", "linear"}:
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_key in {"hist_gradient_boosting", "gradient_boosting", "hgb"}:
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    else:
        raise ValueError("model must be 'logistic' or 'hist_gradient_boosting'.")

    if groups is None:
        warnings.warn(
            "No groups supplied; using stratified random folds. For final results, "
            "group by independent run/simulation production to reduce leakage.",
            RuntimeWarning,
            stacklevel=2,
        )
        splitter = StratifiedKFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(matrix, y)
    else:
        group_array = np.asarray(groups)
        if group_array.shape != y.shape:
            raise ValueError("groups must contain one value per event.")
        splitter = StratifiedGroupKFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(matrix, y, group_array)

    probability = np.full(y.shape, np.nan, dtype=np.float64)
    fold_index = np.full(y.shape, -1, dtype=int)
    for fold, (train_index, test_index) in enumerate(splits):
        fold_estimator = clone(estimator)
        fold_estimator.fit(matrix[train_index], y[train_index])
        probability[test_index] = fold_estimator.predict_proba(matrix[test_index])[:, 1]
        fold_index[test_index] = fold

    if np.any(~np.isfinite(probability)):
        raise RuntimeError("Cross validation did not produce a prediction for every event.")
    metrics = {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
    }
    estimator.fit(matrix, y)
    return ClassifierTrainingResult(
        estimator=estimator,
        feature_names=names,
        out_of_fold_pion_probability=probability,
        fold_index=fold_index,
        metrics=metrics,
    )


def predict_pion_probability(
    training_result: ClassifierTrainingResult,
    feature_rows: Sequence[Mapping[str, float]],
) -> np.ndarray:
    """Evaluate a trained classifier on feature dictionaries."""

    matrix, _ = _feature_matrix(feature_rows, training_result.feature_names)
    return np.asarray(training_result.estimator.predict_proba(matrix)[:, 1], dtype=np.float64)
