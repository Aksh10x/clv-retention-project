"""Stage 1: load, clean, and temporally split the Online Retail II transactions.

Produces, in data/processed/:
    cleaned_transactions.csv   line-item transactions after all cleaning steps
    rfm_calibration.csv        per-customer frequency/recency/T/monetary_value,
                                computed on the calibration period only
                                (lifetimes-library format, used by clv_model.py)
    holdout_actuals.csv        actual per-customer spend during the holdout
                                period, used to validate CLV predictions
    split_config.json          the calibration/holdout cutoff date and the
                                reasoning behind it

Run directly:
    python src/data_prep.py --raw-path data/raw/online_retail_II.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.metrics import log_metrics

# --- Constants (named so every threshold below is explained, not magic) ----

# Invoice numbers prefixed with this letter are cancellations in both the
# original "Online Retail" and "Online Retail II" UCI datasets.
CANCELLED_INVOICE_PREFIX = "C"

# Minimum valid quantity/price: the project spec calls for dropping
# zero/negative quantity or price rows, since these aren't real purchases
# (returns, adjustments, or data errors) and would corrupt RFM/CLV inputs.
MIN_QUANTITY = 1
MIN_PRICE = 0.01

# StockCodes that are documented (UCI dataset description and prior public
# EDA of this dataset) to represent non-product administrative entries
# rather than purchased goods, e.g. postage, discounts, bank charges,
# manual adjustments, and free samples. Kept explicit and logged (see
# `_is_non_product_stock_code`) rather than silently dropped, so the effect
# is visible in outputs/metrics.json instead of hidden.
KNOWN_NON_PRODUCT_STOCK_CODES = frozenset(
    {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK", "S", "AMAZONFEE", "C2", "ADJUST2"}
)

# Real product StockCodes in this dataset are numeric or numeric-with-suffix
# (e.g. "85123A"). A code containing no digits at all is treated as a
# non-product administrative code even if it isn't in the explicit list
# above — this is a heuristic backstop, and every code it catches is logged
# by name so it can be sanity-checked against the actual data rather than
# trusted blindly.
def _is_non_product_stock_code(stock_code: object) -> bool:
    code = str(stock_code).strip().upper()
    return code in KNOWN_NON_PRODUCT_STOCK_CODES or not any(ch.isdigit() for ch in code)


# CLV is predicted over a 6-month horizon per the project spec, so the
# holdout window (used to validate that prediction against real spend)
# is also 6 months.
HOLDOUT_MONTHS = 6

# Column name aliases: "Online Retail" (2015 UCI release) and "Online
# Retail II" (2019 release, used here) use different headers for the same
# fields. Normalizing both to one canonical schema keeps the rest of the
# pipeline dataset-version-agnostic.
_RAW_COLUMN_ALIASES = {
    "invoiceno": "invoice",
    "invoice": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "unitprice": "price",
    "price": "price",
    "customerid": "customer_id",
    "customer id": "customer_id",
    "country": "country",
}

CANONICAL_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map either UCI Online Retail schema onto one canonical column set."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key not in _RAW_COLUMN_ALIASES:
            raise KeyError(
                f"Unrecognized raw column '{col}'. Expected one of the "
                f"'Online Retail' or 'Online Retail II' schemas."
            )
        rename_map[col] = _RAW_COLUMN_ALIASES[key]
    df = df.rename(columns=rename_map)
    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"Raw data is missing required columns: {sorted(missing)}")
    return df[CANONICAL_COLUMNS]


def load_raw_transactions(path: Path) -> pd.DataFrame:
    """Load the raw CSV or Excel file, concatenating all sheets if Excel.

    Online Retail II is distributed as a two-sheet .xlsx (one sheet per
    calendar year); a CSV export may or may not have already merged them.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(path, encoding="ISO-8859-1")

    df = normalize_columns(df)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def _record_step(
    log: list[dict], step: str, before: int, after: int, reason: str, extra: dict | None = None
) -> None:
    entry = {
        "step": step,
        "reason": reason,
        "rows_before": before,
        "rows_dropped": before - after,
        "rows_after": after,
    }
    if extra:
        entry.update(extra)
    log.append(entry)


def clean_transactions(
    df: pd.DataFrame, min_quantity: int = MIN_QUANTITY, min_price: float = MIN_PRICE
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply cleaning steps in order, logging rows dropped and why at each one.

    Order matters for the log to be interpretable: cancelled invoices are
    dropped before the quantity filter, since cancellations are typically
    already recorded as negative quantity — dropping them first means the
    quantity-filter log entry reflects genuinely separate bad rows, not a
    re-count of the same cancellations.
    """
    log: list[dict] = []
    n0 = len(df)

    df = df.dropna(subset=["customer_id"]).copy()
    _record_step(log, "drop_null_customer_id", n0, len(df), "No customer to attribute the purchase to")

    n1 = len(df)
    is_cancelled = df["invoice"].astype(str).str.startswith(CANCELLED_INVOICE_PREFIX)
    df = df.loc[~is_cancelled].copy()
    _record_step(
        log, "drop_cancelled_invoices", n1, len(df),
        f"Invoice prefixed '{CANCELLED_INVOICE_PREFIX}' denotes a cancellation, not a completed purchase",
    )

    n2 = len(df)
    non_product_mask = df["stock_code"].apply(_is_non_product_stock_code)
    dropped_codes = sorted(df.loc[non_product_mask, "stock_code"].astype(str).str.upper().unique().tolist())
    df = df.loc[~non_product_mask].copy()
    _record_step(
        log, "drop_non_product_stock_codes", n2, len(df),
        "Postage/discount/adjustment/fee entries are not purchased goods",
        extra={"stock_codes_dropped": dropped_codes},
    )

    n3 = len(df)
    df = df.loc[df["quantity"] >= min_quantity].copy()
    _record_step(
        log, "drop_low_quantity", n3, len(df),
        f"Quantity < {min_quantity} is a return/adjustment, not a purchase",
    )

    n4 = len(df)
    df = df.loc[df["price"] >= min_price].copy()
    _record_step(
        log, "drop_low_price", n4, len(df),
        f"Price < {min_price} is a data error or non-sale line item",
    )

    return df.reset_index(drop=True), log


def compute_duplicate_stats(df: pd.DataFrame) -> dict:
    """Count exact-duplicate rows in the cleaned transactions (informational only).

    These are NOT dropped: a fully-duplicated row (same invoice, stock
    code, quantity, date, price, and customer) could be a genuine repeat
    line entry or an export artifact, and this dataset gives no way to
    tell the two apart. Deduping would require assuming one or the other,
    so duplicates are logged as a documented characteristic of the data
    rather than treated as a cleaning step.
    """
    n_duplicates = int(df.duplicated().sum())
    return {
        "count": n_duplicates,
        "pct_of_cleaned_rows": float(n_duplicates / len(df)) if len(df) else float("nan"),
    }


def build_invoice_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse line items to one row per (customer, invoice).

    BG/NBD models transactions, not line items — a single order for five
    products must count as one transaction, not five, or frequency will be
    systematically overstated.
    """
    df = df.copy()
    df["line_total"] = df["quantity"] * df["price"]
    invoices = (
        df.groupby(["customer_id", "invoice"], as_index=False)
        .agg(invoice_date=("invoice_date", "min"), invoice_value=("line_total", "sum"))
    )
    return invoices


def temporal_split(
    invoices: pd.DataFrame, holdout_months: int = HOLDOUT_MONTHS
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Split invoice-level transactions into calibration and holdout periods.

    Split by date, not randomly, so calibration-period features never see
    holdout-period information — required for CLV validation to mean
    anything (predicting the future from the future is not a test).
    """
    max_date = invoices["invoice_date"].max()
    calibration_end = max_date - pd.DateOffset(months=holdout_months)

    calibration = invoices.loc[invoices["invoice_date"] <= calibration_end].copy()
    holdout = invoices.loc[invoices["invoice_date"] > calibration_end].copy()
    return calibration_end, calibration, holdout


def compute_rfm_summary(calibration_invoices: pd.DataFrame, calibration_end: pd.Timestamp) -> pd.DataFrame:
    """Per-customer frequency/recency/T/monetary_value over the calibration period.

    Uses the `lifetimes` library's transaction summarizer so the output is
    directly consumable by the BG/NBD and Gamma-Gamma models in
    src/clv_model.py, without a second, potentially inconsistent,
    hand-rolled aggregation.
    """
    from lifetimes.utils import summary_data_from_transaction_data

    summary = summary_data_from_transaction_data(
        calibration_invoices,
        customer_id_col="customer_id",
        datetime_col="invoice_date",
        monetary_value_col="invoice_value",
        observation_period_end=calibration_end,
        freq="D",
    )
    return summary


def compute_holdout_actuals(holdout_invoices: pd.DataFrame) -> pd.DataFrame:
    """Actual total spend per customer during the holdout period (ground truth for CLV validation)."""
    actuals = (
        holdout_invoices.groupby("customer_id")["invoice_value"]
        .sum()
        .rename("holdout_actual_spend")
        .reset_index()
    )
    return actuals


def run(raw_path: Path, output_dir: Path, metrics_path: Path, holdout_months: int = HOLDOUT_MONTHS) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_transactions(raw_path)
    cleaned, cleaning_log = clean_transactions(raw)
    cleaned.to_csv(output_dir / "cleaned_transactions.csv", index=False)
    duplicate_stats = compute_duplicate_stats(cleaned)

    invoices = build_invoice_level(cleaned)
    calibration_end, calibration_invoices, holdout_invoices = temporal_split(invoices, holdout_months)
    calibration_invoices.to_csv(output_dir / "calibration_invoices.csv", index=False)

    rfm = compute_rfm_summary(calibration_invoices, calibration_end)
    rfm.to_csv(output_dir / "rfm_calibration.csv")

    holdout_actuals = compute_holdout_actuals(holdout_invoices)
    holdout_actuals.to_csv(output_dir / "holdout_actuals.csv", index=False)

    split_config = {
        "calibration_end": str(calibration_end.date()),
        "holdout_months": holdout_months,
        "n_customers_calibration": int(calibration_invoices["customer_id"].nunique()),
        "n_customers_holdout": int(holdout_invoices["customer_id"].nunique()),
        "n_invoices_calibration": int(len(calibration_invoices)),
        "n_invoices_holdout": int(len(holdout_invoices)),
    }
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(split_config, f, indent=2)

    log_metrics(
        stage="data_prep",
        metrics={
            "raw_rows": int(len(raw)),
            "cleaned_rows": int(len(cleaned)),
            "date_range": {
                "min": str(cleaned["invoice_date"].min().date()),
                "max": str(cleaned["invoice_date"].max().date()),
            },
            "cleaning_steps": cleaning_log,
            "duplicate_rows": duplicate_stats,
            "split": split_config,
        },
        path=metrics_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load, clean, and temporally split Online Retail II data.")
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw/online_retail_II.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument(
        "--holdout-months", type=int, default=HOLDOUT_MONTHS,
        help="Length of the holdout period in months, matching the CLV prediction horizon.",
    )
    args = parser.parse_args()
    run(args.raw_path, args.output_dir, args.metrics_path, args.holdout_months)


if __name__ == "__main__":
    main()
