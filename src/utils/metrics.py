"""Shared, unit-tested metric functions.

Kept separate from the modeling scripts so both the pipeline code and the
test suite import the exact same implementation — no metric logic lives
only inline in a notebook or a plotting script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average absolute error, in the same units as the target (currency here)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE — penalizes large individual misses more than MAE does."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error_nonzero(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """MAPE computed only over customers with nonzero actual holdout spend.

    MAPE is undefined (division by zero) or explodes for customers whose
    actual holdout spend is 0 — which, in a churn/CLV context, is exactly
    the customers who stopped buying and matters most. Rather than let
    those rows silently blow up the metric, we exclude them from MAPE and
    report separately what fraction of the holdout that exclusion covers,
    so the exclusion itself is visible instead of hidden inside one number.

    Returns:
        (mape, pct_excluded): mape as a fraction (0.25 == 25%), and the
        share of customers excluded because y_true == 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan"), float("nan")

    nonzero_mask = y_true != 0
    pct_excluded = float(1.0 - nonzero_mask.mean())

    if not nonzero_mask.any():
        return float("nan"), pct_excluded

    ape = np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
    return float(np.mean(ape)), pct_excluded


def absolute_percentage_error_values_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-customer APE values (nonzero actuals only), not just a summary statistic.

    Both MAPE and median APE collapse this down to one number each. When
    the two disagree sharply (mean >> median), that's a signal to look at
    the actual distribution — is it a handful of extreme outliers, or a
    systematically fat tail? — before writing up either summary number.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])


def median_absolute_percentage_error_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median (instead of mean) of the same nonzero-actual APE distribution.

    Percentage-error metrics are unbounded above (a small actual with a
    large miss can register as a 10,000% error) but bounded below at 0,
    so the mean is easily dragged far above what a "typical" customer's
    error looks like. Report this alongside MAPE rather than let a mean
    computed on a right-skewed ratio stand alone.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero_mask = y_true != 0
    if not nonzero_mask.any():
        return float("nan")
    ape = np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
    return float(np.median(ape))


def spearman_rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Rank correlation between predicted and actual values.

    Robust to the scale distortions that hit MAPE/MAE when a few
    high-value customers dominate: it only asks whether the model ranks
    customers correctly, which is what a targeting decision actually
    depends on.

    Returns:
        (correlation, p_value)
    """
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = spearmanr(y_true, y_pred)
    return float(result.statistic), float(result.pvalue)


def log_metrics(stage: str, metrics: dict, path: Path) -> None:
    """Merge `metrics` under `stage` into the shared outputs/metrics.json file.

    Each pipeline stage (data_prep, clv_model, churn_model, segmentation)
    calls this independently, so the file accumulates one key per stage
    instead of each stage's run overwriting the others'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    existing[stage] = metrics

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
