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


def compute_decile_lift(y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int = 10):
    """Rank customers by predicted value (descending) into equal-sized deciles,
    and report what share of TOTAL actual spend each decile actually captured.

    Same rank-quality signal as Spearman, expressed as a business metric
    instead of a correlation coefficient: "if you'd targeted only the top
    predicted decile, how much of total actual spend would you have
    captured?" is a more directly actionable number for a targeting
    decision than a correlation coefficient is.

    Customers are split into deciles by POSITION after sorting (via
    np.array_split), not via quantile bin edges on the raw predicted
    values (e.g. pandas.qcut) — the latter throws on duplicate bin edges,
    which is a real risk here: many low-frequency customers can receive
    near-identical predicted CLV (Gamma-Gamma shrinks a sparse customer's
    estimate toward the population mean), so tied predictions are
    expected, not an edge case. Ties are broken by stable sort order
    (original row order), which has no systematic direction, so this
    doesn't inflate the reported lift.

    Returns a pandas DataFrame (imported lazily — this module otherwise
    depends only on numpy/scipy) with one row per decile, decile 1 =
    highest predicted value, ordered so cumulative_pct_of_total_actual_spend
    reaches 1.0 by decile n_deciles.
    """
    import pandas as pd

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        raise ValueError("compute_decile_lift requires at least one customer")

    total_actual = float(y_true.sum())
    order = np.argsort(-y_pred, kind="stable")  # descending by predicted value
    sorted_actual = y_true[order]
    groups = np.array_split(np.arange(y_true.size), n_deciles)

    rows = []
    cumulative = 0.0
    for i, idx in enumerate(groups):
        decile_actual = float(sorted_actual[idx].sum())
        pct = decile_actual / total_actual if total_actual > 0 else float("nan")
        cumulative += pct
        rows.append(
            {
                "decile": i + 1,
                "n_customers": int(idx.size),
                "total_actual_spend": decile_actual,
                "pct_of_total_actual_spend": pct,
                "cumulative_pct_of_total_actual_spend": cumulative,
            }
        )
    return pd.DataFrame(rows)


def top_decile_lift_summary(decile_table, n_deciles: int = 10) -> dict:
    """Top-decile capture rate vs. what a random 10% sample would capture (1/n_deciles), as a ratio.

    A lift of e.g. 3.2x means the top predicted decile captured 3.2 times
    more actual spend than an equal-sized random sample would be expected
    to capture (a random sample captures its population share, 1/n_deciles,
    in expectation, by definition).
    """
    top_pct = float(decile_table.iloc[0]["pct_of_total_actual_spend"])
    random_pct = 1.0 / n_deciles
    return {
        "top_decile_pct_of_actual_spend": top_pct,
        "random_decile_pct_of_actual_spend": random_pct,
        "lift_multiple": top_pct / random_pct if random_pct > 0 else float("nan"),
    }


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
