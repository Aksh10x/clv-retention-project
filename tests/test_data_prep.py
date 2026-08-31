import pandas as pd
import pytest

from data_prep import (
    build_invoice_level,
    clean_transactions,
    compute_duplicate_stats,
    compute_holdout_actuals,
    normalize_columns,
    temporal_split,
)


def test_normalize_columns_online_retail_ii_schema():
    df = pd.DataFrame(
        columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    )
    normalized = normalize_columns(df)
    assert list(normalized.columns) == [
        "invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country",
    ]


def test_normalize_columns_original_online_retail_schema():
    df = pd.DataFrame(
        columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    )
    normalized = normalize_columns(df)
    assert list(normalized.columns) == [
        "invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country",
    ]


def test_normalize_columns_raises_on_unrecognized_column():
    df = pd.DataFrame(columns=["Invoice", "StockCode", "SomeUnknownColumn"])
    with pytest.raises(KeyError):
        normalize_columns(df)


def _sample_transactions() -> pd.DataFrame:
    # Each row below is designed to be caught by exactly one cleaning step,
    # so the per-step drop counts in the tests are unambiguous:
    #   row 2 -> cancelled invoice ('C' prefix)
    #   row 3 -> non-product stock code ('POST'), valid customer/invoice
    #   row 5 -> zero quantity
    #   row 6 -> null customer_id (price is also invalid, but customer_id
    #            is dropped first so it never reaches the price filter)
    return pd.DataFrame(
        {
            "invoice": ["536365", "536365", "C536366", "536367", "536368", "536369", "536370"],
            "stock_code": ["85123A", "85123A", "22423", "POST", "84406B", "22112", "22112"],
            "description": ["desc"] * 7,
            "quantity": [6, 6, -1, 1, 3, 0, -5],
            "invoice_date": pd.to_datetime(
                ["2011-01-01"] * 2 + ["2011-01-02"] + ["2011-01-03"] * 4
            ),
            "price": [2.55, 2.55, 1.85, 18.0, 4.25, 3.0, -1.0],
            "customer_id": [17850.0, 17850.0, 17850.0, 17850.0, 13047.0, 13047.0, None],
            "country": ["United Kingdom"] * 7,
        }
    )


def test_clean_transactions_drops_null_customer_id():
    df = _sample_transactions()
    cleaned, log = clean_transactions(df)
    assert cleaned["customer_id"].notna().all()
    step = next(s for s in log if s["step"] == "drop_null_customer_id")
    assert step["rows_dropped"] == 1


def test_clean_transactions_drops_cancelled_invoices():
    df = _sample_transactions()
    cleaned, _ = clean_transactions(df)
    assert not cleaned["invoice"].astype(str).str.startswith("C").any()


def test_clean_transactions_drops_non_product_stock_codes():
    df = _sample_transactions()
    cleaned, log = clean_transactions(df)
    assert "POST" not in cleaned["stock_code"].values
    step = next(s for s in log if s["step"] == "drop_non_product_stock_codes")
    assert "POST" in step["stock_codes_dropped"]


def test_clean_transactions_drops_non_positive_quantity_and_price():
    df = _sample_transactions()
    cleaned, _ = clean_transactions(df)
    assert (cleaned["quantity"] >= 1).all()
    assert (cleaned["price"] >= 0.01).all()


def test_clean_transactions_keeps_valid_rows():
    df = _sample_transactions()
    cleaned, _ = clean_transactions(df)
    # only the two 85123A lines for customer 17850 and the 84406B line for
    # 13047 should survive every filter
    assert len(cleaned) == 3
    assert set(cleaned["customer_id"]) == {17850.0, 13047.0}


def test_clean_transactions_log_accounts_for_every_dropped_row():
    df = _sample_transactions()
    cleaned, log = clean_transactions(df)
    total_dropped = sum(step["rows_dropped"] for step in log)
    assert len(df) - len(cleaned) == total_dropped


def test_compute_duplicate_stats_counts_exact_duplicates_without_dropping():
    df = pd.DataFrame(
        {
            "invoice": ["1", "1", "2"],
            "stock_code": ["A", "A", "B"],
            "quantity": [1, 1, 2],
        }
    )
    stats = compute_duplicate_stats(df)
    assert stats["count"] == 1
    assert stats["pct_of_cleaned_rows"] == pytest.approx(1 / 3)
    assert len(df) == 3  # unmodified — duplicates are logged, not dropped


def test_build_invoice_level_sums_line_items_per_invoice():
    df = pd.DataFrame(
        {
            "invoice": ["1", "1", "2"],
            "stock_code": ["A", "B", "C"],
            "description": ["d"] * 3,
            "quantity": [2, 3, 1],
            "invoice_date": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-01-05"]),
            "price": [10.0, 5.0, 20.0],
            "customer_id": [1.0, 1.0, 1.0],
            "country": ["UK"] * 3,
        }
    )
    invoices = build_invoice_level(df)
    assert len(invoices) == 2  # collapsed to one row per invoice
    invoice_1 = invoices.loc[invoices["invoice"] == "1"].iloc[0]
    assert invoice_1["invoice_value"] == pytest.approx(2 * 10.0 + 3 * 5.0)


def test_temporal_split_respects_cutoff_boundary():
    invoices = pd.DataFrame(
        {
            "customer_id": [1, 1, 1],
            "invoice": ["a", "b", "c"],
            "invoice_date": pd.to_datetime(["2011-01-01", "2011-04-01", "2011-07-01"]),
            "invoice_value": [10.0, 20.0, 30.0],
        }
    )
    calibration_end, calibration, holdout = temporal_split(invoices, holdout_months=3)
    assert calibration_end == pd.Timestamp("2011-04-01")
    # calibration_end itself is inclusive of calibration
    assert set(calibration["invoice"]) == {"a", "b"}
    assert set(holdout["invoice"]) == {"c"}


def test_compute_holdout_actuals_sums_per_customer():
    holdout = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "invoice": ["a", "b", "c"],
            "invoice_value": [10.0, 15.0, 40.0],
        }
    )
    actuals = compute_holdout_actuals(holdout)
    result = dict(zip(actuals["customer_id"], actuals["holdout_actual_spend"]))
    assert result == {1: 25.0, 2: 40.0}
