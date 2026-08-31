import numpy as np
import pandas as pd
import pytest

from clv_model import (
    check_independence,
    compute_clv_decile_lift,
    fit_bgnbd,
    fit_gamma_gamma,
    identify_worst_predictions,
    predict_clv,
    validate_predictions,
)


def test_check_independence_excludes_one_time_buyers():
    rfm = pd.DataFrame(
        {
            "frequency": [0, 0, 3, 5],
            "monetary_value": [0, 0, 100, 500],
        }
    )
    result = check_independence(rfm)
    assert result["n_repeat_customers"] == 2  # only frequency > 0 rows counted


def test_check_independence_flags_correlation_above_threshold():
    rng = np.random.default_rng(42)
    frequency = rng.integers(1, 20, size=200).astype(float)
    monetary_value = frequency * 10 + rng.normal(0, 1, size=200)  # strongly correlated by construction
    rfm = pd.DataFrame({"frequency": frequency, "monetary_value": monetary_value})

    result = check_independence(rfm, threshold=0.3)
    assert result["exceeds_threshold"] is True
    assert result["pearson_correlation"] > 0.3


def test_check_independence_passes_uncorrelated_data():
    rng = np.random.default_rng(7)
    frequency = rng.integers(1, 20, size=200).astype(float)
    monetary_value = rng.normal(100, 10, size=200)  # independent of frequency by construction
    rfm = pd.DataFrame({"frequency": frequency, "monetary_value": monetary_value})

    result = check_independence(rfm, threshold=0.3)
    assert result["exceeds_threshold"] is False


def test_validate_predictions_fills_missing_holdout_with_zero():
    predictions = pd.DataFrame({"customer_id": [1, 2, 3], "predicted_clv": [100.0, 200.0, 50.0]})
    holdout_actuals = pd.DataFrame({"customer_id": [1, 3], "holdout_actual_spend": [90.0, 40.0]})
    # customer 2 made no holdout purchase at all -> absent from holdout_actuals, not zero-valued

    merged, metrics = validate_predictions(predictions, holdout_actuals)

    row2 = merged.loc[merged["customer_id"] == 2].iloc[0]
    assert row2["holdout_actual_spend"] == 0.0
    assert metrics["n_customers"] == 3


def test_validate_predictions_metrics_match_manual_calc():
    predictions = pd.DataFrame({"customer_id": [1, 2], "predicted_clv": [110.0, 180.0]})
    holdout_actuals = pd.DataFrame({"customer_id": [1, 2], "holdout_actual_spend": [100.0, 200.0]})

    _, metrics = validate_predictions(predictions, holdout_actuals)

    assert metrics["mae"] == pytest.approx((10 + 20) / 2)
    assert metrics["mape_nonzero_actuals"] == pytest.approx((0.10 + 0.10) / 2)


def test_compute_clv_decile_lift_includes_zero_actual_churned_customers():
    # unlike the APE-based metrics, decile lift must NOT exclude churned
    # (zero-actual) customers -- their $0 spend is real signal for total
    # spend captured, not an undefined ratio.
    predictions = pd.DataFrame(
        {"customer_id": range(1, 21), "predicted_clv": list(range(20, 0, -1))}  # descending predicted value
    )
    holdout_actuals = pd.DataFrame(
        {"customer_id": range(1, 11), "holdout_actual_spend": [100.0] * 10}  # only half of customers spent
    )
    merged, _ = validate_predictions(predictions, holdout_actuals)

    decile_table, summary = compute_clv_decile_lift(merged, n_deciles=10)

    assert decile_table["n_customers"].sum() == 20  # all 20 customers included, not just the 10 spenders
    # top predicted decile = customers 1-2 (highest predicted_clv, 20 and 19),
    # who are also among the actual spenders (customer_id 1-10) -> should
    # capture 2/10 = 20% of total actual spend (1000), i.e. above the 10% random rate
    assert summary["top_decile_pct_of_actual_spend"] == pytest.approx(0.20)
    assert summary["lift_multiple"] == pytest.approx(2.0)


def test_identify_worst_predictions_ranks_by_ape_descending():
    predictions = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "frequency": [5.0, 2.0, 1.0, 0.0],
            "recency": [100.0, 50.0, 30.0, 0.0],
            "T": [200.0, 150.0, 100.0, 100.0],
            "monetary_value": [80.0, 90.0, 95.0, 0.0],
            "predicted_clv": [500.0, 110.0, 105.0, 200.0],
        }
    )
    holdout_actuals = pd.DataFrame(
        {"customer_id": [1, 2, 3, 4], "holdout_actual_spend": [5.0, 100.0, 100.0, 0.0]}
    )
    merged, _ = validate_predictions(predictions, holdout_actuals)

    worst = identify_worst_predictions(merged, top_n=2)

    assert list(worst["customer_id"]) == [1, 2]  # customer 4 (zero actual) excluded, ordered by ape desc
    assert worst.iloc[0]["ape"] > worst.iloc[1]["ape"]


@pytest.fixture
def synthetic_rfm() -> pd.DataFrame:
    """Frequency/recency/T drawn from lifetimes' own BG/NBD generator.

    Hand-rolled uncorrelated random noise for these columns produced a
    degenerate likelihood surface that failed to converge (real customer
    behavior has internal structure a point-process model expects; pure
    noise doesn't). Sampling from the model's own generative process
    guarantees a fittable, if synthetic, dataset.

    Note: `beta_geometric_nbd_model` draws from NumPy's legacy global RNG
    (`np.random.beta`/`gamma`/`exponential`), not a `Generator` instance —
    seeding a local `np.random.default_rng(...)` has no effect on it. Must
    seed the legacy global API directly, or this fixture's convergence
    becomes dependent on what other tests ran first and consumed that
    shared global state (this is exactly how it turned up as a flaky
    failure — passed in isolation, intermittently failed in the full suite).
    """
    from lifetimes.generate_data import beta_geometric_nbd_model

    np.random.seed(0)
    rng = np.random.default_rng(0)
    n = 300
    data = beta_geometric_nbd_model(T=500, r=0.8, alpha=50, a=1, b=3, size=n)
    data = data.reset_index(drop=True)

    monetary_value = np.where(
        data["frequency"] > 0, rng.gamma(shape=4.0, scale=40.0, size=n) + 5, 0.0
    )
    return pd.DataFrame(
        {
            "customer_id": range(n),
            "frequency": data["frequency"],
            "recency": data["recency"],
            "T": data["T"],
            "monetary_value": monetary_value,
        }
    )


def test_fit_and_predict_smoke(synthetic_rfm):
    """End-to-end fit + predict on synthetic data: no exceptions, sane output shape."""
    bgf = fit_bgnbd(synthetic_rfm)
    ggf = fit_gamma_gamma(synthetic_rfm)

    predictions = predict_clv(bgf, ggf, synthetic_rfm, time_months=6, discount_rate=0.0)

    assert len(predictions) == len(synthetic_rfm)
    assert predictions["predicted_clv"].notna().all()
    assert (predictions["predicted_clv"] >= 0).all()
    # one-time/never-buyers (frequency == 0) must still get a prediction,
    # not be dropped or produce NaN
    zero_freq_customers = predictions.loc[synthetic_rfm["frequency"] == 0]
    assert len(zero_freq_customers) > 0
    assert zero_freq_customers["predicted_clv"].notna().all()
