import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn_model import (
    NON_RECENCY_FEATURES,
    build_features,
    compute_inter_purchase_interval_std,
    compute_leakage_diagnostics,
    compute_purchase_trend,
    determine_churn_threshold,
    evaluate_model,
    label_churn,
)


def _invoices(rows: list[tuple]) -> pd.DataFrame:
    """rows: (customer_id, invoice, date_str, value)"""
    return pd.DataFrame(
        rows, columns=["customer_id", "invoice", "invoice_date", "invoice_value"]
    ).assign(invoice_date=lambda d: pd.to_datetime(d["invoice_date"]))


def test_purchase_trend_all_spend_in_second_half_is_positive_one():
    # calibration window spans 2011-01-01..2011-03-01, midpoint ~2011-01-30
    invoices = _invoices([(1, "a", "2011-02-25", 100.0)])
    trend = compute_purchase_trend(
        pd.concat([invoices, _invoices([(2, "z", "2011-01-05", 50.0)])])
    )
    assert trend.loc[1] == pytest.approx(1.0)


def test_purchase_trend_all_spend_in_first_half_is_negative_one():
    invoices = pd.concat(
        [
            _invoices([(1, "a", "2011-01-05", 100.0)]),
            _invoices([(2, "z", "2011-02-25", 50.0)]),
        ]
    )
    trend = compute_purchase_trend(invoices)
    assert trend.loc[1] == pytest.approx(-1.0)


def test_purchase_trend_balanced_spend_is_zero():
    invoices = _invoices(
        [
            (1, "a", "2011-01-05", 100.0),
            (1, "b", "2011-02-25", 100.0),
        ]
    )
    trend = compute_purchase_trend(invoices)
    assert trend.loc[1] == pytest.approx(0.0)


def test_inter_purchase_interval_std_single_invoice_is_nan():
    invoices = _invoices([(1, "a", "2011-01-01", 100.0)])
    std = compute_inter_purchase_interval_std(invoices)
    assert np.isnan(std.loc[1])


def test_inter_purchase_interval_std_two_invoices_is_zero():
    # exactly one gap observed -> population std of a single value is 0, not NaN
    invoices = _invoices([(1, "a", "2011-01-01", 100.0), (1, "b", "2011-01-11", 100.0)])
    std = compute_inter_purchase_interval_std(invoices)
    assert std.loc[1] == pytest.approx(0.0)


def test_inter_purchase_interval_std_varying_gaps():
    # gaps of 10 and 30 days
    invoices = _invoices(
        [
            (1, "a", "2011-01-01", 100.0),
            (1, "b", "2011-01-11", 100.0),
            (1, "c", "2011-02-10", 100.0),
        ]
    )
    std = compute_inter_purchase_interval_std(invoices)
    assert std.loc[1] == pytest.approx(np.std([10.0, 30.0], ddof=0))


def test_determine_churn_threshold_matches_numpy_percentile():
    invoices = _invoices(
        [(1, "a", "2011-01-01", 10.0), (1, "b", "2011-01-11", 10.0), (1, "c", "2011-02-10", 10.0)]
    )
    threshold, gaps = determine_churn_threshold(invoices, percentile=50)
    assert threshold == pytest.approx(np.percentile([10.0, 30.0], 50))
    assert sorted(gaps.tolist()) == [10.0, 30.0]


def test_label_churn_applies_threshold_correctly():
    features = pd.DataFrame({"customer_id": [1, 2, 3], "days_since_last_purchase": [50.0, 119.0, 200.0]})
    labeled = label_churn(features, threshold=119.0)
    assert list(labeled["churned"]) == [0, 0, 1]  # strictly greater than threshold


def test_build_features_merges_rfm_and_behavioral_signals():
    rfm = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "frequency": [2.0, 0.0],
            "recency": [40.0, 0.0],
            "T": [60.0, 30.0],
            "monetary_value": [80.0, 0.0],
        }
    )
    invoices = _invoices(
        [
            (1, "a", "2011-01-01", 100.0),
            (1, "b", "2011-01-20", 100.0),
            (1, "c", "2011-02-10", 100.0),
            (2, "d", "2011-01-15", 50.0),
        ]
    )
    features = build_features(rfm, invoices, calibration_end=pd.Timestamp("2011-03-01"))

    # customer 1: T - recency = 60 - 40 = 20 days since last purchase
    row1 = features.loc[features["customer_id"] == 1].iloc[0]
    assert row1["days_since_last_purchase"] == pytest.approx(20.0)
    assert not pd.isna(row1["inter_purchase_interval_std"])  # 2 gaps observed

    # customer 2: single invoice, falling in the first half of the global
    # calibration window (2011-01-01..2011-02-10, midpoint ~01-21) -> trend -1.
    # interval std is left NaN (a single invoice has zero observed gaps).
    row2 = features.loc[features["customer_id"] == 2].iloc[0]
    assert row2["purchase_trend"] == pytest.approx(-1.0)
    assert pd.isna(row2["inter_purchase_interval_std"])


def test_compute_leakage_diagnostics_counts_short_tenure_customers_correctly():
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {
            "frequency": rng.integers(0, 10, size=n).astype(float),
            "T": rng.uniform(0, 400, size=n),
            "monetary_value": rng.uniform(0, 500, size=n),
            "purchase_trend": rng.uniform(-1, 1, size=n),
            "inter_purchase_interval_std": rng.uniform(0, 50, size=n),
        }
    )
    # label loosely related to T so both classes appear in a split
    y = (df["T"] > df["T"].median()).astype(int)
    X_train, X_test = df.iloc[:150], df.iloc[150:]
    y_train, y_test = y.iloc[:150], y.iloc[150:]

    threshold = 119.0
    result = compute_leakage_diagnostics(X_train, y_train, X_test, y_test, threshold)

    expected_short_tenure = int((df["T"] < threshold).sum())
    assert result["short_tenure_mechanical_constraint"]["n_customers_with_t_below_threshold"] == expected_short_tenure
    assert set(result["single_feature_auc"].keys()) == set(NON_RECENCY_FEATURES)
    assert 0.0 <= result["non_recency_auc_without_purchase_trend"] <= 1.0


def test_evaluate_model_confusion_matrix_and_auc():
    X = pd.DataFrame({"x": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]})
    y = pd.Series([0, 0, 1, 1, 1, 0])
    model = LogisticRegression().fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"]["tp"] + metrics["confusion_matrix"]["fn"] == int(y.sum())
    assert metrics["confusion_matrix"]["tn"] + metrics["confusion_matrix"]["fp"] == int((y == 0).sum())
