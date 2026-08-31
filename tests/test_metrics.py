import json

import numpy as np
import pytest

from utils.metrics import (
    absolute_percentage_error_values_nonzero,
    log_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error_nonzero,
    median_absolute_percentage_error_nonzero,
    root_mean_squared_error,
    spearman_rank_correlation,
)


def test_mean_absolute_error_basic():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    assert np.isclose(mean_absolute_error(y_true, y_pred), (2 + 2 + 3) / 3)


def test_mean_absolute_error_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert mean_absolute_error(y, y) == 0.0


def test_root_mean_squared_error_penalizes_large_errors_more_than_mae():
    y_true = np.array([0.0, 0.0])
    y_pred_small_errors = np.array([1.0, 1.0])
    y_pred_one_big_error = np.array([0.0, 2.0])

    rmse_small = root_mean_squared_error(y_true, y_pred_small_errors)
    rmse_big = root_mean_squared_error(y_true, y_pred_one_big_error)

    # same total absolute error (2.0), but RMSE should be higher when it's
    # concentrated in one large miss rather than spread evenly
    assert mean_absolute_error(y_true, y_pred_small_errors) == mean_absolute_error(
        y_true, y_pred_one_big_error
    )
    assert rmse_big > rmse_small


def test_mape_excludes_zero_actuals_and_reports_exclusion_rate():
    y_true = np.array([0.0, 0.0, 100.0, 200.0])
    y_pred = np.array([5.0, 10.0, 110.0, 180.0])

    mape, pct_excluded = mean_absolute_percentage_error_nonzero(y_true, y_pred)

    # only the two nonzero-actual rows contribute: |110-100|/100=0.10, |180-200|/200=0.10
    assert np.isclose(mape, 0.10)
    assert np.isclose(pct_excluded, 0.5)


def test_mape_all_zero_actuals_returns_nan_mape_full_exclusion():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([5.0, 10.0])

    mape, pct_excluded = mean_absolute_percentage_error_nonzero(y_true, y_pred)

    assert np.isnan(mape)
    assert pct_excluded == 1.0


def test_median_ape_is_robust_to_a_single_extreme_outlier():
    y_true = np.array([100.0, 100.0, 100.0, 1.0])
    y_pred = np.array([110.0, 90.0, 105.0, 500.0])  # last row: 100x actual, dominates the mean

    mape, _ = mean_absolute_percentage_error_nonzero(y_true, y_pred)
    median_ape = median_absolute_percentage_error_nonzero(y_true, y_pred)

    assert median_ape < mape  # the typical error is much smaller than the mean suggests
    assert median_ape == pytest.approx(0.10)  # median of [0.10, 0.10, 0.05, 499.0]


def test_ape_values_excludes_zero_actuals_and_matches_summary_stats():
    y_true = np.array([0.0, 100.0, 200.0, 50.0])
    y_pred = np.array([5.0, 110.0, 180.0, 200.0])

    ape_values = absolute_percentage_error_values_nonzero(y_true, y_pred)

    assert len(ape_values) == 3  # the zero-actual row is excluded
    mape, _ = mean_absolute_percentage_error_nonzero(y_true, y_pred)
    median_ape = median_absolute_percentage_error_nonzero(y_true, y_pred)
    assert np.isclose(ape_values.mean(), mape)
    assert np.isclose(np.median(ape_values), median_ape)


def test_median_ape_all_zero_actuals_returns_nan():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([5.0, 10.0])
    assert np.isnan(median_absolute_percentage_error_nonzero(y_true, y_pred))


def test_spearman_perfect_rank_correlation():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([10, 20, 30, 40, 50])  # same ranking, different scale

    corr, _ = spearman_rank_correlation(y_true, y_pred)
    assert np.isclose(corr, 1.0)


def test_spearman_inverse_rank_correlation():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])

    corr, _ = spearman_rank_correlation(y_true, y_pred)
    assert np.isclose(corr, -1.0)


def test_log_metrics_merges_stages_without_clobbering(tmp_path):
    path = tmp_path / "metrics.json"

    log_metrics("data_prep", {"raw_rows": 100}, path)
    log_metrics("clv_model", {"mape": 0.2}, path)

    with open(path) as f:
        result = json.load(f)

    assert result["data_prep"] == {"raw_rows": 100}
    assert result["clv_model"] == {"mape": 0.2}


def test_log_metrics_overwrites_same_stage_on_rerun(tmp_path):
    path = tmp_path / "metrics.json"

    log_metrics("data_prep", {"raw_rows": 100}, path)
    log_metrics("data_prep", {"raw_rows": 200}, path)

    with open(path) as f:
        result = json.load(f)

    assert result["data_prep"] == {"raw_rows": 200}
