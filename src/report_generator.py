"""Stage 5: automated PDF retention report from the outputs of stages 1-4.

Replaces an interactive dashboard with a static, shareable report: loads
CLV predictions, churn probabilities, and cohort assignments produced by
clv_model.py / churn_model.py / segmentation.py, plus the metrics.json
each stage logs, and renders a single PDF via Jinja2 + WeasyPrint.

Every number in the report is read from the pipeline's own saved outputs
— nothing here re-derives or hardcodes a result that a prior stage
already computed and validated.

Run directly:
    python src/report_generator.py --run-name 2025-Q4

macOS setup: WeasyPrint needs Pango (`brew install pango`) — this module
sets DYLD_FALLBACK_LIBRARY_PATH itself before importing weasyprint so no
manual env var is needed, but the Homebrew package must be installed.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# See module docstring: must be set before `import weasyprint`, since it
# resolves Pango/GObject via dlopen at import time, and Homebrew on
# Apple Silicon installs to /opt/homebrew rather than a default search path.
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
import weasyprint  # noqa: E402

# --- Constants ---------------------------------------------------------

REPORT_TITLE = "CLV & Retention Intelligence Report"
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.html.jinja"

TOP_N_WATCHLIST = 20

# Illustrative campaign-simulation assumptions (see report's own callout
# box — these are stated, not measured, and the report says so).
CAMPAIGN_N_TARGETS = 500  # ~10% of the ~4,942-customer base: one illustrative outreach batch
CAMPAIGN_COST_PER_OUTREACH = 5.0  # GBP, e.g. an email/direct-mail retention offer
CAMPAIGN_RETENTION_LIFT_PP = 0.15  # assumed churn-probability reduction for a contacted customer

# Recommended action per cohort, keyed by (value_tier, risk_tier) parsed
# from the cohort's own label — not by cluster id, so this still makes
# sense if a future run produces a different number/composition of
# cohorts (any combination not covered falls back to a generic message).
COHORT_ACTIONS = {
    ("High-Value", "Low-Risk"): (
        "Protect and grow. Low churn urgency — prioritize loyalty treatment, "
        "cross-sell, and proactive relationship management over reactive retention spend."
    ),
    ("High-Value", "At-Risk"): (
        "Highest retention priority. This cohort carries the largest expected value at "
        "risk — personal outreach and a tailored retention offer are justified even at "
        "meaningful per-customer cost."
    ),
    ("Low-Value", "Low-Risk"): (
        "Low-touch maintenance. Standard lifecycle marketing is sufficient; not a "
        "priority for dedicated retention spend given low value at stake."
    ),
    ("Low-Value", "At-Risk"): (
        "Scalable, low-cost win-back only. Broad automated re-engagement (e.g. an "
        "email offer) may be worthwhile in aggregate, but individually-targeted "
        "retention spend is unlikely to be justified by the CLV at stake."
    ),
}
WATCHLIST_ACTIONS = {
    ("High-Value", "At-Risk"): "Immediate personal outreach",
    ("High-Value", "Low-Risk"): "Monitor; loyalty touch",
    ("Low-Value", "At-Risk"): "Low-cost automated win-back",
    ("Low-Value", "Low-Risk"): "No action needed",
}


def file_uri(path: Path) -> str:
    return Path(path).resolve().as_uri()


def load_all_outputs(processed_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    segments_path = output_dir / "segments.csv"
    metrics_path = output_dir / "metrics.json"
    split_config_path = processed_dir / "split_config.json"

    for path, stage in [
        (segments_path, "segmentation.py"),
        (metrics_path, "any pipeline stage"),
        (split_config_path, "data_prep.py"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run {stage} first.")

    segments = pd.read_csv(segments_path)
    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(split_config_path) as f:
        split_config = json.load(f)

    for stage in ("data_prep", "clv_model", "churn_model", "segmentation"):
        if stage not in metrics:
            raise KeyError(f"metrics.json has no '{stage}' section — run src/{stage}.py first.")

    return segments, metrics, split_config


def build_cohort_table(segments: pd.DataFrame) -> list[dict]:
    total_clv = segments["predicted_clv"].sum()
    n_total = len(segments)

    rows = []
    for label, group in segments.groupby("segment_label"):
        value_tier, risk_tier = (part.strip() for part in label.split("/"))
        rows.append(
            {
                "name": label,
                "n_customers": len(group),
                "pct_of_customers": len(group) / n_total,
                "avg_clv": float(group["predicted_clv"].mean()),
                "avg_churn_risk": float(group["predicted_churn_proba"].mean()),
                "pct_of_portfolio_value": float(group["predicted_clv"].sum() / total_clv),
                "recommended_action": COHORT_ACTIONS.get(
                    (value_tier, risk_tier), "Review this cohort's centroid values manually."
                ),
            }
        )
    rows.sort(key=lambda r: r["avg_clv"], reverse=True)
    return rows


def build_watchlist(segments: pd.DataFrame, top_n: int = TOP_N_WATCHLIST) -> list[dict]:
    """Top N customers by expected value at risk, with a per-customer (not per-cluster) action.

    Deliberately does NOT reuse each customer's `segment_label`: that's
    their K-Means *cluster's* label, derived from the cluster's mean CLV
    and mean churn probability — not this individual's own values. A
    customer can belong to the "Low-Risk" cluster (low mean churn
    probability) while individually having a 97% churn probability
    themselves (that's exactly why they're on this watchlist in the
    first place). Using the cluster label here silently mislabeled every
    high-risk watchlist customer as "Low-Risk" on the first pass — caught
    by reading the actual rendered PDF, where a customer with a 96.6%
    churn probability was shown getting a "Low-Risk" action.
    """
    from churn_model import CLASSIFICATION_THRESHOLD

    df = segments.copy()
    df["value_at_risk"] = df["predicted_clv"] * df["predicted_churn_proba"]
    clv_median = df["predicted_clv"].median()
    top = df.nlargest(top_n, "value_at_risk")

    rows = []
    for _, row in top.iterrows():
        value_tier = "High-Value" if row["predicted_clv"] >= clv_median else "Low-Value"
        risk_tier = "At-Risk" if row["predicted_churn_proba"] >= CLASSIFICATION_THRESHOLD else "Low-Risk"
        rows.append(
            {
                "customer_id": int(row["customer_id"]),
                "clv": float(row["predicted_clv"]),
                "churn_proba": float(row["predicted_churn_proba"]),
                "recency": float(row["days_since_last_purchase"]),
                "value_at_risk": float(row["value_at_risk"]),
                "recommended_action": WATCHLIST_ACTIONS.get((value_tier, risk_tier), "Review manually"),
            }
        )
    return rows


def simulate_campaign_roi(segments: pd.DataFrame) -> dict:
    n = min(CAMPAIGN_N_TARGETS, len(segments))
    df = segments.copy()
    df["value_at_risk"] = df["predicted_clv"] * df["predicted_churn_proba"]

    recommended = df.nlargest(n, "value_at_risk")
    cost = n * CAMPAIGN_COST_PER_OUTREACH

    recommended_value_saved = float(recommended["predicted_clv"].sum() * CAMPAIGN_RETENTION_LIFT_PP)
    # random targeting's expected value saved, computed analytically as
    # the population mean CLV -- this IS the expectation of a random
    # N-sample, so no need to actually draw random samples.
    random_value_saved = float(df["predicted_clv"].mean() * n * CAMPAIGN_RETENTION_LIFT_PP)

    def _summarize(value_saved: float) -> dict:
        return {
            "cost": cost,
            "value_saved": value_saved,
            "net_roi": value_saved - cost,
            "roi_multiple": value_saved / cost,
        }

    return {
        "n_targets": n,
        "pct_of_base": n / len(segments),
        "cost_per_outreach": CAMPAIGN_COST_PER_OUTREACH,
        "retention_lift": CAMPAIGN_RETENTION_LIFT_PP,
        "recommended": _summarize(recommended_value_saved),
        "random": _summarize(random_value_saved),
        "uplift_vs_random": recommended_value_saved - random_value_saved,
    }


def build_clv_validation(metrics: dict, output_dir: Path) -> dict:
    v = metrics["clv_model"]["validation"]
    decile = metrics["clv_model"]["decile_lift"]
    return {
        "mae": v["mae"],
        "rmse": v["rmse"],
        "mape": v["mape_nonzero_actuals"],
        "median_ape": v["median_ape_nonzero_actuals"],
        "spearman": v["spearman_correlation"],
        "mape_note": (
            "Mean MAPE is pulled well above the typical customer's error by a small number of "
            "customers whose spend changed sharply between the calibration and holdout periods "
            "(e.g. a previously active customer who nearly stopped purchasing). Median APE and "
            "Spearman rank correlation are the more representative numbers for a targeting use "
            "case, where ranking customers correctly matters more than exact dollar forecasts."
        ),
        "scatter_chart_path": file_uri(output_dir / "plots" / "clv_predicted_vs_actual.png"),
        "top_decile_pct_of_spend": decile["top_decile_pct_of_actual_spend"],
        "decile_lift_multiple": decile["lift_multiple"],
        "decile_chart_path": file_uri(output_dir / "plots" / "clv_decile_lift.png"),
    }


def build_churn_validation(metrics: dict, output_dir: Path) -> dict:
    cm = metrics["churn_model"]
    shap_ranking = sorted(
        ({"feature": k, "value": v} for k, v in cm["mean_abs_shap"].items()),
        key=lambda d: -d["value"],
    )
    leak = cm["leakage_diagnostics"]
    single = leak["single_feature_auc"]
    max_feature = max(single, key=single.get)

    return {
        "recency_only_auc": cm["recency_only_baseline"]["roc_auc"],
        "non_recency_auc": cm["non_recency_model"]["roc_auc"],
        "full_auc": cm["full_model"]["roc_auc"],
        "churn_threshold_days": cm["churn_threshold_days"],
        "shap_ranking": shap_ranking,
        "shap_chart_path": file_uri(output_dir / "plots" / "churn_shap_summary.png"),
        "leakage": {
            "pct_short_tenure": leak["short_tenure_mechanical_constraint"]["pct_customers_with_t_below_threshold"],
            "n_short_tenure": leak["short_tenure_mechanical_constraint"]["n_customers_with_t_below_threshold"],
            "t_solo_auc": single["T"],
            "no_trend_auc": leak["non_recency_auc_without_purchase_trend"],
            "max_single_feature_auc": single[max_feature],
            "max_single_feature_name": max_feature,
        },
    }


def build_segmentation_validation(metrics: dict, segments: pd.DataFrame, output_dir: Path) -> dict:
    sm = metrics["segmentation"]
    chosen_k = sm["chosen_k"]
    clv_churn_correlation = float(
        np.log1p(segments["predicted_clv"]).corr(segments["predicted_churn_proba"])
    )
    return {
        "chosen_k": chosen_k,
        "silhouette_at_k": sm["silhouette_scores"][str(chosen_k)],
        "rfm_only_ari": sm["rfm_only_stability_ari"],
        "clv_churn_correlation": clv_churn_correlation,
        "k_selection_chart_path": file_uri(output_dir / "plots" / "segmentation_k_selection.png"),
    }


def build_methodology(metrics: dict, split_config: dict) -> dict:
    dp = metrics["data_prep"]
    null_step = next(s for s in dp["cleaning_steps"] if s["step"] == "drop_null_customer_id")
    return {
        "date_range": f"{dp['date_range']['min']} to {dp['date_range']['max']}",
        "raw_rows": dp["raw_rows"],
        "cleaned_rows": dp["cleaned_rows"],
        "calibration_end": split_config["calibration_end"],
        "cleaning_steps": dp["cleaning_steps"],
        "null_customer_pct": null_step["rows_dropped"] / dp["raw_rows"],
        "duplicate_pct": dp["duplicate_rows"]["pct_of_cleaned_rows"],
        "duplicate_count": dp["duplicate_rows"]["count"],
    }


def build_narrative(
    n_customers: int, total_clv: float, churn_rate: float, cohorts: list[dict], clv_churn_correlation: float
) -> str:
    """Template-based paragraph — sentence structure is fixed, every number is computed."""
    highest_value = max(cohorts, key=lambda c: c["avg_clv"])
    highest_risk = max(cohorts, key=lambda c: c["avg_churn_risk"])
    cohort_names = " and ".join(c["name"] for c in cohorts)

    return (
        f"This report analyzes {n_customers:,} customers with a combined predicted 6-month CLV of "
        f"£{total_clv:,.0f}. Based on a recency-derived threshold, {churn_rate:.1%} of the base is "
        f"currently classified as churned. Segmentation converged to {len(cohorts)} cohorts — "
        f"{cohort_names} — reflecting a {clv_churn_correlation:.2f} correlation between predicted "
        f"value and churn probability in this customer base, rather than four independently-varying "
        f"risk/value combinations. The {highest_risk['name']} cohort carries the highest average "
        f"churn risk ({highest_risk['avg_churn_risk']:.1%}) but represents "
        f"{highest_risk['pct_of_portfolio_value']:.1%} of total portfolio value, while the "
        f"{highest_value['name']} cohort — {highest_value['pct_of_customers']:.1%} of customers — "
        f"holds {highest_value['pct_of_portfolio_value']:.1%} of portfolio value at comparatively low risk."
    )


def plot_cohort_value_bar(segments: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = segments.groupby("segment_label")["predicted_clv"].sum().sort_values(ascending=False)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(totals)))

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.bar(totals.index, totals.values, color=colors)
    ax.set_ylabel("Total predicted 6-month CLV (£)")
    ax.set_title("Portfolio value by cohort")
    for i, v in enumerate(totals.values):
        ax.text(i, v, f"£{v:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_html(context: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    env.filters["int_fmt"] = lambda v: f"{int(round(v)):,}"
    env.filters["money_fmt"] = lambda v: f"{v:,.0f}"
    env.filters["pct_fmt"] = lambda v: f"{v * 100:.1f}%"
    env.filters["mult_fmt"] = lambda v: f"{v:.2f}x"
    template = env.get_template(TEMPLATE_NAME)
    return template.render(**context)


def render_pdf(html: str, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weasyprint.HTML(string=html).write_pdf(str(output_path))


def generate_report(processed_dir: Path, output_dir: Path, run_name: str) -> Path:
    output_dir = Path(output_dir)
    segments, metrics, split_config = load_all_outputs(processed_dir, output_dir)

    reports_dir = output_dir / "reports"
    assets_dir = reports_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    cohort_bar_path = assets_dir / "cohort_value_bar.png"
    plot_cohort_value_bar(segments, cohort_bar_path)

    cohorts = build_cohort_table(segments)
    segmentation_validation = build_segmentation_validation(metrics, segments, output_dir)

    n_customers = len(segments)
    total_portfolio_clv = float(segments["predicted_clv"].sum())
    overall_churn_rate = float(metrics["churn_model"]["churn_rate"])

    context = {
        "report_title": REPORT_TITLE,
        "run_name": run_name,
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "n_customers": n_customers,
        "total_portfolio_clv": total_portfolio_clv,
        "overall_churn_rate": overall_churn_rate,
        "narrative_summary": build_narrative(
            n_customers, total_portfolio_clv, overall_churn_rate, cohorts,
            segmentation_validation["clv_churn_correlation"],
        ),
        "cohorts": cohorts,
        "cohort_bar_chart_path": file_uri(cohort_bar_path),
        "watchlist": build_watchlist(segments),
        "clv_validation": build_clv_validation(metrics, output_dir),
        "churn_validation": build_churn_validation(metrics, output_dir),
        "segmentation_validation": segmentation_validation,
        "independence_check": {
            "correlation": metrics["clv_model"]["independence_check"]["pearson_correlation"],
            "threshold": metrics["clv_model"]["independence_check"]["warning_threshold"],
        },
        "campaign_roi": simulate_campaign_roi(segments),
        "methodology": build_methodology(metrics, split_config),
    }

    html = render_html(context)
    pdf_path = reports_dir / f"retention_report_{run_name}.pdf"
    render_pdf(html, pdf_path)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the PDF retention report from pipeline outputs.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    pdf_path = generate_report(args.processed_dir, args.output_dir, args.run_name)
    print(f"Report written to {pdf_path}")


if __name__ == "__main__":
    main()
