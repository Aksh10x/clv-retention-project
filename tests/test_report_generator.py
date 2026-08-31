import json

import numpy as np
import pandas as pd
import pytest

from report_generator import (
    CAMPAIGN_COST_PER_OUTREACH,
    CAMPAIGN_RETENTION_LIFT_PP,
    build_cohort_table,
    build_narrative,
    build_watchlist,
    file_uri,
    generate_report,
    simulate_campaign_roi,
)


def _sample_segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "segment_label": [
                "High-Value / Low-Risk", "High-Value / Low-Risk",
                "Low-Value / At-Risk", "Low-Value / At-Risk", "Low-Value / At-Risk",
            ],
            "predicted_clv": [1000.0, 2000.0, 50.0, 100.0, 30.0],
            "predicted_churn_proba": [0.1, 0.05, 0.9, 0.95, 0.99],
            "frequency": [5, 8, 0, 1, 0],
            "days_since_last_purchase": [10, 20, 400, 300, 500],
            "total_calibration_spend": [1200.0, 2500.0, 60.0, 150.0, 40.0],
        }
    )


def test_build_cohort_table_aggregates_and_sorts_by_clv_descending():
    cohorts = build_cohort_table(_sample_segments())

    assert len(cohorts) == 2
    assert cohorts[0]["name"] == "High-Value / Low-Risk"  # highest avg_clv sorts first
    assert cohorts[0]["n_customers"] == 2
    assert cohorts[0]["avg_clv"] == pytest.approx(1500.0)
    assert cohorts[1]["avg_clv"] == pytest.approx(60.0)


def test_build_cohort_table_portfolio_value_percentages_sum_to_one():
    cohorts = build_cohort_table(_sample_segments())
    assert sum(c["pct_of_portfolio_value"] for c in cohorts) == pytest.approx(1.0)


def test_build_cohort_table_action_derived_from_parsed_tiers_not_hardcoded():
    cohorts = build_cohort_table(_sample_segments())
    high_value_low_risk = next(c for c in cohorts if c["name"] == "High-Value / Low-Risk")
    low_value_at_risk = next(c for c in cohorts if c["name"] == "Low-Value / At-Risk")
    assert "protect" in high_value_low_risk["recommended_action"].lower()
    assert "win-back" in low_value_at_risk["recommended_action"].lower()


def test_build_cohort_table_unknown_tier_combo_falls_back_gracefully():
    df = _sample_segments().copy()
    df["segment_label"] = "Medium-Value / Unclear-Risk"
    cohorts = build_cohort_table(df)
    assert "review" in cohorts[0]["recommended_action"].lower()


def test_build_watchlist_sorts_by_value_at_risk_descending():
    watchlist = build_watchlist(_sample_segments(), top_n=5)
    values_at_risk = [w["value_at_risk"] for w in watchlist]
    assert values_at_risk == sorted(values_at_risk, reverse=True)
    # value_at_risk = clv * churn_proba: customer 1 (1000*0.1=100) and
    # customer 2 (2000*0.05=100) tie for highest; customer 4 (100*0.95=95) is next
    assert watchlist[0]["value_at_risk"] == pytest.approx(100.0)
    assert watchlist[2]["customer_id"] == 4


def test_build_watchlist_action_uses_individual_values_not_cluster_label():
    # customer belongs to a "Low-Risk" cluster (cluster mean churn is low)
    # but is INDIVIDUALLY at 97% churn probability -- exactly the outlier
    # case that got mislabeled by reusing the cluster's segment_label.
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "segment_label": ["High-Value / Low-Risk", "High-Value / Low-Risk", "High-Value / Low-Risk"],
            "predicted_clv": [1000.0, 1200.0, 1500.0],
            "predicted_churn_proba": [0.05, 0.1, 0.97],  # customer 3 is the individual outlier
            "frequency": [5, 6, 7],
            "days_since_last_purchase": [10, 20, 15],
            "total_calibration_spend": [1000.0, 1200.0, 1500.0],
        }
    )
    watchlist = build_watchlist(df, top_n=3)
    outlier = next(w for w in watchlist if w["customer_id"] == 3)
    assert outlier["recommended_action"] == "Immediate personal outreach"


def test_build_watchlist_respects_top_n():
    watchlist = build_watchlist(_sample_segments(), top_n=2)
    assert len(watchlist) == 2


def test_simulate_campaign_roi_arithmetic():
    segments = _sample_segments()
    result = simulate_campaign_roi(segments)

    n = result["n_targets"]
    assert n == len(segments)  # fixture is smaller than CAMPAIGN_N_TARGETS, so capped at population size
    expected_cost = n * CAMPAIGN_COST_PER_OUTREACH
    assert result["recommended"]["cost"] == pytest.approx(expected_cost)

    # recommended targeting picks the highest value-at-risk customers,
    # so its total CLV (and thus value saved) must be >= random's expectation
    assert result["recommended"]["value_saved"] >= result["random"]["value_saved"]
    assert result["uplift_vs_random"] >= 0


def test_simulate_campaign_roi_random_uses_population_mean_analytically():
    segments = _sample_segments()
    result = simulate_campaign_roi(segments)
    n = result["n_targets"]
    expected_random_value_saved = segments["predicted_clv"].mean() * n * CAMPAIGN_RETENTION_LIFT_PP
    assert result["random"]["value_saved"] == pytest.approx(expected_random_value_saved)


def test_build_narrative_includes_computed_numbers_not_placeholders():
    cohorts = build_cohort_table(_sample_segments())
    narrative = build_narrative(
        n_customers=5, total_clv=3180.0, churn_rate=0.6, cohorts=cohorts, clv_churn_correlation=-0.67
    )
    assert "5" in narrative
    assert "60.0%" in narrative
    assert "-0.67" in narrative
    assert "High-Value / Low-Risk" in narrative
    assert "{{" not in narrative and "}}" not in narrative  # no unrendered template artifacts


def test_file_uri_produces_a_file_scheme_url(tmp_path):
    p = tmp_path / "chart.png"
    p.write_bytes(b"fake png bytes")
    uri = file_uri(p)
    assert uri.startswith("file://")
    assert "chart.png" in uri


@pytest.fixture
def synthetic_pipeline_outputs(tmp_path):
    """Builds a minimal but complete outputs/ + data/processed/ tree so
    generate_report() can run truly end-to-end, including real PDF
    rendering, without depending on the real (large) pipeline outputs.
    """
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    plots_dir = output_dir / "plots"
    processed_dir.mkdir()
    plots_dir.mkdir(parents=True)

    _sample_segments().to_csv(output_dir / "segments.csv", index=False)

    # tiny placeholder PNGs for every chart the template embeds
    import matplotlib.pyplot as plt

    for name in [
        "clv_predicted_vs_actual.png", "clv_decile_lift.png",
        "churn_shap_summary.png", "segmentation_k_selection.png",
    ]:
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        fig.savefig(plots_dir / name)
        plt.close(fig)

    metrics = {
        "data_prep": {
            "raw_rows": 1000,
            "cleaned_rows": 800,
            "date_range": {"min": "2020-01-01", "max": "2021-01-01"},
            "cleaning_steps": [
                {"step": "drop_null_customer_id", "reason": "no customer", "rows_dropped": 150, "rows_before": 1000, "rows_after": 850},
                {"step": "drop_cancelled_invoices", "reason": "cancelled", "rows_dropped": 50, "rows_before": 850, "rows_after": 800},
            ],
            "duplicate_rows": {"count": 20, "pct_of_cleaned_rows": 0.025},
        },
        "clv_model": {
            "validation": {
                "mae": 100.0, "rmse": 500.0, "mape_nonzero_actuals": 0.5,
                "median_ape_nonzero_actuals": 0.3, "spearman_correlation": 0.6,
            },
            "independence_check": {"pearson_correlation": 0.1, "warning_threshold": 0.3},
            "decile_lift": {
                "top_decile_pct_of_actual_spend": 0.55, "random_decile_pct_of_actual_spend": 0.1,
                "lift_multiple": 5.5,
            },
        },
        "churn_model": {
            "churn_rate": 0.5,
            "churn_threshold_days": 100,
            "recency_only_baseline": {"roc_auc": 1.0},
            "non_recency_model": {"roc_auc": 0.9},
            "full_model": {"roc_auc": 1.0},
            "mean_abs_shap": {"days_since_last_purchase": 4.0, "frequency": 0.5},
            "leakage_diagnostics": {
                "single_feature_auc": {"frequency": 0.7, "T": 0.5, "monetary_value": 0.6, "purchase_trend": 0.6, "inter_purchase_interval_std": 0.5},
                "non_recency_auc_without_purchase_trend": 0.88,
                "short_tenure_mechanical_constraint": {"pct_customers_with_t_below_threshold": 0.1, "n_customers_with_t_below_threshold": 50},
            },
        },
        "segmentation": {
            "chosen_k": 2,
            "silhouette_scores": {"2": 0.46, "3": 0.43},
            "rfm_only_stability_ari": 0.42,
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    with open(processed_dir / "split_config.json", "w") as f:
        json.dump({"calibration_end": "2021-01-01"}, f)

    return processed_dir, output_dir


def test_generate_report_end_to_end_produces_a_real_pdf(synthetic_pipeline_outputs):
    processed_dir, output_dir = synthetic_pipeline_outputs
    pdf_path = generate_report(processed_dir, output_dir, run_name="test-run")

    assert pdf_path.exists()
    assert pdf_path.suffix == ".pdf"
    assert pdf_path.stat().st_size > 5_000  # a real multi-page PDF, not an empty/error stub
    with open(pdf_path, "rb") as f:
        assert f.read(5) == b"%PDF-"
