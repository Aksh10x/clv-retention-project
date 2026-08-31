import numpy as np
import pandas as pd
import pytest

from segmentation import (
    compute_rfm_only_stability,
    compute_segment_profiles,
    label_clusters,
    load_inputs,
    prepare_feature_matrix,
    select_k,
)


def test_prepare_feature_matrix_log_transforms_skewed_columns_only():
    df = pd.DataFrame(
        {
            "predicted_clv": [0.0, 100.0, 10000.0],
            "predicted_churn_proba": [0.1, 0.5, 0.9],
            "frequency": [0.0, 5.0, 50.0],
            "days_since_last_purchase": [10.0, 100.0, 400.0],
            "total_calibration_spend": [10.0, 500.0, 50000.0],
        }
    )
    X, scaler = prepare_feature_matrix(df, list(df.columns))

    # after log1p + standardize, the extreme raw skew (10000x) should be
    # compressed rather than dominating the scaled matrix
    clv_col = X[:, df.columns.get_loc("predicted_clv")]
    assert clv_col.max() < 10  # not the raw ~10000x spread anymore
    # standardized output: each column has ~zero mean
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)


def test_prepare_feature_matrix_does_not_log_transform_churn_proba():
    df = pd.DataFrame({"predicted_churn_proba": [0.0, 0.5, 1.0]})
    X, _ = prepare_feature_matrix(df, ["predicted_churn_proba"])
    # standardized raw values (not log1p'd) -> symmetric around the middle value
    assert X[1, 0] == pytest.approx(0.0, abs=1e-8)


def test_label_clusters_derives_labels_from_actual_centroids_not_position():
    # cluster 5 (an arbitrary, non-sequential id) is high value/high risk;
    # cluster 0 is low value/low risk. Labels must reflect that regardless
    # of which cluster id happens to be first.
    df = pd.DataFrame(
        {
            "cluster": [5, 5, 5, 0, 0, 0],
            "predicted_clv": [1000.0, 1200.0, 900.0, 10.0, 20.0, 15.0],
            "predicted_churn_proba": [0.9, 0.85, 0.95, 0.1, 0.05, 0.15],
        }
    )
    labels = label_clusters(df)
    assert labels[5] == "High-Value / At-Risk"
    assert labels[0] == "Low-Value / Low-Risk"


def test_compute_segment_profiles_aggregates_correctly_and_sums_to_all_customers():
    df = pd.DataFrame(
        {
            "cluster": [0, 0, 1],
            "segment_label": ["A", "A", "B"],
            "customer_id": [1, 2, 3],
            "predicted_clv": [100.0, 200.0, 50.0],
            "predicted_churn_proba": [0.2, 0.4, 0.8],
            "frequency": [2.0, 4.0, 0.0],
            "days_since_last_purchase": [10.0, 20.0, 300.0],
            "total_calibration_spend": [100.0, 300.0, 20.0],
        }
    )
    profiles = compute_segment_profiles(df)

    assert profiles["n_customers"].sum() == 3
    assert profiles["pct_of_customers"].sum() == pytest.approx(1.0)
    cluster_a = profiles.loc[profiles["cluster"] == 0].iloc[0]
    assert cluster_a["mean_predicted_clv"] == pytest.approx(150.0)
    assert cluster_a["label"] == "A"


def test_select_k_finds_well_separated_synthetic_clusters():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    cluster_b = rng.normal(loc=[10, 10], scale=0.3, size=(30, 2))
    cluster_c = rng.normal(loc=[10, 0], scale=0.3, size=(30, 2))
    X = np.vstack([cluster_a, cluster_b, cluster_c])

    inertias, silhouettes, best_k = select_k(X, k_candidates=[2, 3, 4, 5])

    assert best_k == 3  # matches the true number of generated clusters
    assert inertias[2] > inertias[5]  # inertia decreases monotonically as K grows


def test_rfm_only_stability_perfect_agreement_when_features_are_identical():
    # if the "full" and "rfm-only" feature sets were literally the same
    # columns, the two clusterings must agree perfectly (ARI == 1.0) --
    # a sanity check on the ARI wiring itself, not the real feature sets.
    rng = np.random.default_rng(1)
    cluster_a = rng.normal(loc=[0, 0, 0], scale=0.2, size=(20, 3))
    cluster_b = rng.normal(loc=[8, 8, 8], scale=0.2, size=(20, 3))
    X = np.vstack([cluster_a, cluster_b])

    from segmentation import RFM_ONLY_FEATURES
    df = pd.DataFrame(X, columns=RFM_ONLY_FEATURES)
    from sklearn.cluster import KMeans
    full_labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X).labels_

    _, ari = compute_rfm_only_stability(df, full_labels, k=2)
    assert ari == pytest.approx(1.0)


def test_load_inputs_raises_on_row_mismatch(tmp_path):
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    processed_dir.mkdir()
    output_dir.mkdir()

    pd.DataFrame({"customer_id": [1, 2], "predicted_clv": [10.0, 20.0]}).to_csv(
        output_dir / "clv_predictions.csv", index=False
    )
    # churn_predictions only has customer 1 -> merge should lose customer 2
    pd.DataFrame(
        {"customer_id": [1], "frequency": [2.0], "days_since_last_purchase": [5.0], "predicted_churn_proba": [0.3]}
    ).to_csv(output_dir / "churn_predictions.csv", index=False)
    pd.DataFrame({"customer_id": [1, 2], "invoice": ["a", "b"], "invoice_value": [10.0, 20.0]}).to_csv(
        processed_dir / "calibration_invoices.csv", index=False
    )

    with pytest.raises(ValueError):
        load_inputs(processed_dir, output_dir)
