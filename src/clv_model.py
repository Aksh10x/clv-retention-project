"""Stage 2: BG/NBD + Gamma-Gamma customer lifetime value model.

Fits BG/NBD (expected future transactions) and Gamma-Gamma (expected
monetary value per transaction) on the calibration-period RFM table
produced by data_prep.py, predicts 6-month CLV per customer, and
validates the prediction against actual holdout-period spend.

Produces, in outputs/:
    clv_predictions.csv        per-customer prediction + actual + errors
    plots/clv_predicted_vs_actual.png
    models/bgnbd.pkl, models/gamma_gamma.pkl

Run directly:
    python src/clv_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from scipy.stats import pearsonr

from utils.metrics import (
    absolute_percentage_error_values_nonzero,
    compute_decile_lift,
    log_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error_nonzero,
    median_absolute_percentage_error_nonzero,
    root_mean_squared_error,
    spearman_rank_correlation,
    top_decile_lift_summary,
)

# How many of the worst-APE customers to save for manual inspection —
# enough to spot a pattern (a handful of outliers vs. a systematic skew
# across dozens of customers) without dumping the whole dataset.
N_WORST_PREDICTIONS_TO_SAVE = 20

# Standard business convention for a targeting/lift analysis; deciles
# (not quintiles or ventiles) are the usual granularity for this kind of
# "would targeting the top N% have worked" business metric.
N_DECILES = 10

matplotlib.use("Agg")  # headless: this script never opens an interactive window

# --- Constants ---------------------------------------------------------

# Prediction horizon, per the project spec ("predict 6-month CLV").
# `lifetimes.GammaGammaFitter.customer_lifetime_value` takes this in months
# and internally assumes a ~30-day month; the calibration/holdout split
# built in data_prep.py uses calendar months (~183 days for 6 calendar
# months), so the two are a close but not exact match — a known, minor
# approximation, not worth hand-tuning for a ~1-day discrepancy.
CLV_HORIZON_MONTHS = 6

# Kept at 0 (no discounting) specifically for validation: the holdout
# actuals from data_prep.py are raw, undiscounted spend, so discounting
# the prediction would introduce a systematic downward bias before the
# comparison even starts, making MAPE/MAE look worse than the model
# actually is. A nonzero rate is still exposed via --discount-rate for
# business-facing CLV figures (e.g. the dashboard), where discounting the
# future is the point.
VALIDATION_DISCOUNT_RATE = 0.0

# Decision: if the Pearson correlation between frequency and average
# monetary value (on repeat customers) exceeds this, log it as a
# documented limitation but proceed with Gamma-Gamma regardless — mild
# violations of the independence assumption are standard practice to
# tolerate, not a reason to block the pipeline.
INDEPENDENCE_CORR_WARNING_THRESHOLD = 0.3

# lifetimes default: no L2 penalty on the MLE fit. Only worth raising
# above 0 if a fit fails to converge or visibly overfits a sparse
# customer base, neither of which applies here.
BGNBD_PENALIZER_COEF = 0.0
GAMMA_GAMMA_PENALIZER_COEF = 0.0


def load_processed_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    processed_dir = Path(processed_dir)
    rfm = pd.read_csv(processed_dir / "rfm_calibration.csv")
    holdout_actuals = pd.read_csv(processed_dir / "holdout_actuals.csv")
    with open(processed_dir / "split_config.json") as f:
        split_config = json.load(f)
    return rfm, holdout_actuals, split_config


def check_independence(rfm: pd.DataFrame, threshold: float = INDEPENDENCE_CORR_WARNING_THRESHOLD) -> dict:
    """Pearson correlation between frequency and monetary value, repeat customers only.

    Gamma-Gamma assumes these are independent. one-time buyers (frequency
    == 0) have no repeat-purchase monetary value to speak of and are
    excluded from the check, matching the population Gamma-Gamma is
    actually fit on.
    """
    repeat = rfm.loc[rfm["frequency"] > 0]
    corr, p_value = pearsonr(repeat["frequency"], repeat["monetary_value"])

    exceeds_threshold = bool(abs(corr) > threshold)
    result = {
        "pearson_correlation": float(corr),
        "p_value": float(p_value),
        "n_repeat_customers": int(len(repeat)),
        "warning_threshold": threshold,
        "exceeds_threshold": exceeds_threshold,
    }
    if exceeds_threshold:
        print(
            f"WARNING: frequency/monetary_value correlation ({corr:.3f}) exceeds "
            f"the {threshold} threshold — Gamma-Gamma's independence assumption is "
            f"violated more than mildly. Proceeding anyway per project convention; "
            f"treat CLV estimates as a documented limitation, not invalidated."
        )
    else:
        print(f"Independence check OK: frequency/monetary_value correlation = {corr:.3f}")
    return result


def fit_bgnbd(rfm: pd.DataFrame) -> BetaGeoFitter:
    bgf = BetaGeoFitter(penalizer_coef=BGNBD_PENALIZER_COEF)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    return bgf


def fit_gamma_gamma(rfm: pd.DataFrame) -> GammaGammaFitter:
    """Fit on repeat customers only — Gamma-Gamma requires frequency > 0."""
    repeat = rfm.loc[rfm["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=GAMMA_GAMMA_PENALIZER_COEF)
    ggf.fit(repeat["frequency"], repeat["monetary_value"])
    return ggf


def predict_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    rfm: pd.DataFrame,
    time_months: int = CLV_HORIZON_MONTHS,
    discount_rate: float = VALIDATION_DISCOUNT_RATE,
) -> pd.DataFrame:
    """Predict expected transactions, avg order value, and CLV for every customer.

    One-time buyers (frequency == 0) are included: Gamma-Gamma's average
    profit estimate for them shrinks to the population mean (its weight on
    an individual customer's own monetary value depends on their observed
    frequency, which is 0), and BG/NBD handles frequency == 0 natively.
    """
    predicted_transactions = bgf.conditional_expected_number_of_purchases_up_to_time(
        time_months * 30, rfm["frequency"], rfm["recency"], rfm["T"]
    )
    predicted_avg_order_value = ggf.conditional_expected_average_profit(
        rfm["frequency"], rfm["monetary_value"]
    )
    predicted_clv = ggf.customer_lifetime_value(
        bgf,
        rfm["frequency"],
        rfm["recency"],
        rfm["T"],
        rfm["monetary_value"],
        time=time_months,
        freq="D",
        discount_rate=discount_rate,
    )

    result = rfm[["customer_id", "frequency", "recency", "T", "monetary_value"]].copy()
    result["predicted_transactions"] = predicted_transactions.values
    result["predicted_avg_order_value"] = predicted_avg_order_value.values
    result["predicted_clv"] = predicted_clv.values
    return result


def validate_predictions(predictions: pd.DataFrame, holdout_actuals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join predictions to holdout actuals and compute validation metrics.

    Calibration customers absent from holdout_actuals made no purchase in
    the holdout period — their actual spend is 0, not missing, and they
    are kept in the comparison (that's exactly the churn case CLV needs
    to get right, not a row to drop).
    """
    merged = predictions.merge(holdout_actuals, on="customer_id", how="left")
    merged["holdout_actual_spend"] = merged["holdout_actual_spend"].fillna(0.0)
    merged["abs_error"] = (merged["predicted_clv"] - merged["holdout_actual_spend"]).abs()

    y_true = merged["holdout_actual_spend"].to_numpy()
    y_pred = merged["predicted_clv"].to_numpy()

    mape, pct_excluded_from_mape = mean_absolute_percentage_error_nonzero(y_true, y_pred)
    median_ape = median_absolute_percentage_error_nonzero(y_true, y_pred)
    spearman_corr, spearman_p = spearman_rank_correlation(y_true, y_pred)
    ape_values = absolute_percentage_error_values_nonzero(y_true, y_pred)

    # per-row ape for downstream outlier inspection/plotting; NaN for
    # zero-actual (churned) customers, where APE is undefined
    merged["ape"] = np.nan
    merged.loc[merged["holdout_actual_spend"] != 0, "ape"] = ape_values

    metrics = {
        "n_customers": int(len(merged)),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape_nonzero_actuals": mape,
        "median_ape_nonzero_actuals": median_ape,
        "pct_customers_excluded_from_mape": pct_excluded_from_mape,
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "ape_percentiles": {
            f"p{p}": float(np.percentile(ape_values, p)) for p in (10, 25, 50, 75, 90, 95, 99)
        },
    }
    return merged, metrics


def compute_clv_decile_lift(merged: pd.DataFrame, n_deciles: int = N_DECILES) -> tuple[pd.DataFrame, dict]:
    """Decile-lift analysis: same rank-quality signal as Spearman, expressed
    as a business metric — what % of total actual holdout spend would
    targeting only the top predicted decile have captured, vs. the 1/n_deciles
    a random sample of the same size would capture in expectation.

    Uses the full customer population (including zero-actual/churned
    customers), unlike the APE-based metrics above — a decile-lift
    analysis is about total spend captured, and a churned customer's $0
    actual spend is real information for that (not an undefined ratio to
    exclude, the way it is for percentage-error metrics).
    """
    decile_table = compute_decile_lift(
        merged["holdout_actual_spend"].to_numpy(), merged["predicted_clv"].to_numpy(), n_deciles=n_deciles
    )
    summary = top_decile_lift_summary(decile_table, n_deciles=n_deciles)
    return decile_table, summary


def plot_decile_lift(decile_table: pd.DataFrame, summary: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_deciles = len(decile_table)
    random_pct = summary["random_decile_pct_of_actual_spend"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        decile_table["decile"], decile_table["pct_of_total_actual_spend"] * 100,
        color="steelblue", edgecolor="white",
    )
    bars[0].set_color("crimson")  # highlight the top decile — the one the headline number is about
    ax.axhline(random_pct * 100, color="gray", linestyle="--", label=f"random {n_deciles}-way split ({random_pct:.0%})")
    ax.set_xlabel("Decile (1 = highest predicted CLV)")
    ax.set_ylabel("% of total actual holdout spend captured")
    ax.set_title("CLV decile lift: predicted-value ranking vs. random targeting")
    ax.set_xticks(decile_table["decile"])
    ax.legend()
    ax.text(
        0.98, 0.95,
        f"Top decile: {summary['top_decile_pct_of_actual_spend']:.1%} of spend "
        f"({summary['lift_multiple']:.1f}x random)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def identify_worst_predictions(merged: pd.DataFrame, top_n: int = N_WORST_PREDICTIONS_TO_SAVE) -> pd.DataFrame:
    """The top_n customers with the largest absolute percentage error, for manual inspection.

    Distinguishes "a few extreme outliers dragging the mean" (this table
    looks wildly different from the median customer) from "a systematic
    model problem" (this table looks like an ordinary, larger version of
    the median case) — a judgment call a single summary statistic can't
    make on its own.
    """
    worst = merged.dropna(subset=["ape"]).nlargest(top_n, "ape")
    return worst[
        ["customer_id", "frequency", "recency", "T", "monetary_value",
         "predicted_clv", "holdout_actual_spend", "ape"]
    ].reset_index(drop=True)


def plot_ape_distribution(merged: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ape = merged["ape"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    # log-scale bins: APE ranges from ~0 to >100x for a few customers,
    # linear bins would make the whole distribution invisible under the tail
    bins = np.logspace(np.log10(max(ape.min(), 1e-3)), np.log10(ape.max()), 40)
    ax.hist(ape, bins=bins, color="steelblue", edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(metrics["mape_nonzero_actuals"], color="crimson", linestyle="--", label="mean (MAPE)")
    ax.axvline(metrics["median_ape_nonzero_actuals"], color="darkorange", linestyle="--", label="median")
    ax.set_xlabel("Absolute percentage error (log scale)")
    ax.set_ylabel("Number of customers")
    ax.set_title("Distribution of CLV prediction error (nonzero-actual customers)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(merged: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # log1p so zero-actual (churned) customers are visible instead of
    # collapsing onto an unplottable log(0) point
    x = merged["holdout_actual_spend"].clip(lower=0)
    y = merged["predicted_clv"].clip(lower=0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.3, s=12, edgecolor="none")
    max_val = max(x.max(), y.max())
    ax.plot([0, max_val], [0, max_val], color="crimson", linestyle="--", label="perfect prediction")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_xlabel("Actual holdout-period spend (£)")
    ax.set_ylabel(f"Predicted {CLV_HORIZON_MONTHS}-month CLV (£)")
    ax.set_title("CLV: predicted vs. actual holdout spend")
    ax.legend(loc="upper left")
    ax.text(
        0.98, 0.02,
        f"MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}\n"
        f"MAPE(nonzero)={metrics['mape_nonzero_actuals']:.1%} "
        f"/ median APE={metrics['median_ape_nonzero_actuals']:.1%} "
        f"(excl. {metrics['pct_customers_excluded_from_mape']:.1%})\n"
        f"Spearman r={metrics['spearman_correlation']:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run(
    processed_dir: Path,
    output_dir: Path,
    metrics_path: Path,
    time_months: int = CLV_HORIZON_MONTHS,
    discount_rate: float = VALIDATION_DISCOUNT_RATE,
    independence_threshold: float = INDEPENDENCE_CORR_WARNING_THRESHOLD,
) -> None:
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rfm, holdout_actuals, split_config = load_processed_data(processed_dir)

    independence_result = check_independence(rfm, independence_threshold)

    bgf = fit_bgnbd(rfm)
    ggf = fit_gamma_gamma(rfm)
    bgf.save_model(str(models_dir / "bgnbd.pkl"))
    ggf.save_model(str(models_dir / "gamma_gamma.pkl"))

    predictions = predict_clv(bgf, ggf, rfm, time_months, discount_rate)
    merged, validation_metrics = validate_predictions(predictions, holdout_actuals)
    merged.to_csv(output_dir / "clv_predictions.csv", index=False)

    plot_predicted_vs_actual(merged, validation_metrics, plots_dir / "clv_predicted_vs_actual.png")
    plot_ape_distribution(merged, validation_metrics, plots_dir / "clv_ape_distribution.png")

    worst_predictions = identify_worst_predictions(merged)
    worst_predictions.to_csv(output_dir / "clv_worst_predictions.csv", index=False)

    decile_table, decile_lift_summary = compute_clv_decile_lift(merged)
    decile_table.to_csv(output_dir / "clv_decile_lift.csv", index=False)
    plot_decile_lift(decile_table, decile_lift_summary, plots_dir / "clv_decile_lift.png")

    log_metrics(
        stage="clv_model",
        metrics={
            "calibration_customers": int(len(rfm)),
            "one_time_buyers": int((rfm["frequency"] == 0).sum()),
            "independence_check": independence_result,
            "bgnbd_params": {k: float(v) for k, v in bgf.params_.to_dict().items()},
            "gamma_gamma_params": {k: float(v) for k, v in ggf.params_.to_dict().items()},
            "clv_horizon_months": time_months,
            "validation_discount_rate": discount_rate,
            "calibration_end": split_config["calibration_end"],
            "validation": validation_metrics,
            "decile_lift": decile_lift_summary,
        },
        path=metrics_path,
    )

    print(f"Independence correlation: {independence_result['pearson_correlation']:.3f}")
    print(f"MAE={validation_metrics['mae']:.2f}  RMSE={validation_metrics['rmse']:.2f}  "
          f"MAPE(nonzero)={validation_metrics['mape_nonzero_actuals']:.1%}  "
          f"median APE={validation_metrics['median_ape_nonzero_actuals']:.1%}  "
          f"Spearman r={validation_metrics['spearman_correlation']:.3f}")
    print(f"Top-decile capture: {decile_lift_summary['top_decile_pct_of_actual_spend']:.1%} of total actual "
          f"holdout spend ({decile_lift_summary['lift_multiple']:.2f}x vs. random "
          f"{decile_lift_summary['random_decile_pct_of_actual_spend']:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit BG/NBD + Gamma-Gamma and predict/validate CLV.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--time-months", type=int, default=CLV_HORIZON_MONTHS)
    parser.add_argument(
        "--discount-rate", type=float, default=VALIDATION_DISCOUNT_RATE,
        help="Monthly discount rate for the DCF CLV calc. Default 0.0 keeps predictions "
             "undiscounted so they're comparable to raw holdout actuals during validation.",
    )
    parser.add_argument("--independence-threshold", type=float, default=INDEPENDENCE_CORR_WARNING_THRESHOLD)
    args = parser.parse_args()
    run(
        args.processed_dir, args.output_dir, args.metrics_path,
        args.time_months, args.discount_rate, args.independence_threshold,
    )


if __name__ == "__main__":
    main()
