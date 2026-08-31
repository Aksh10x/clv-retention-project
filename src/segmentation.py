"""Stage 4: K-Means customer segmentation on CLV + churn risk + RFM.

Combines predictions from clv_model.py and churn_model.py with total
calibration-period spend into a feature matrix, picks K via elbow +
silhouette, fits K-Means, and derives cohort labels from actual centroid
characteristics (never hardcoded).

Also runs an RFM-only stability check (see module docstring section
below) as a sanity comparison, not a pass/fail gate.

Produces, in outputs/:
    segments.csv                    per-customer cluster assignment + label
    segment_profiles.csv            per-cluster summary table
    plots/segmentation_k_selection.png
    plots/segmentation_scatter.png
    models/kmeans.pkl, models/segmentation_scaler.pkl

Run directly:
    python src/segmentation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from churn_model import CLASSIFICATION_THRESHOLD
from utils.metrics import log_metrics

matplotlib.use("Agg")

# --- Constants -------------------------------------------------------------

RANDOM_STATE = 42
N_INIT = 10  # pinned explicitly rather than relying on sklearn's version-dependent 'auto' default

# K values to evaluate via elbow + silhouette. Capped at 8: beyond that,
# cohorts stop being distinct enough to act on for a targeting use case.
K_CANDIDATES = list(range(2, 9))

# CLV, frequency, and spend are all non-negative and heavily right-skewed
# (a handful of large-account customers dominate the raw scale — see the
# CLV validation writeup). Log-transforming before standardizing prevents
# K-Means' Euclidean distance from being driven almost entirely by
# whoever has the single largest CLV/spend value. Churn probability is
# already bounded [0, 1] and roughly symmetric, so it's excluded here.
LOG_TRANSFORM_FEATURES = ["predicted_clv", "frequency", "total_calibration_spend"]

FULL_FEATURES = ["predicted_clv", "predicted_churn_proba", "frequency", "days_since_last_purchase", "total_calibration_spend"]

# RFM-only stability check: same customers, same K as the full model (so
# the comparison isolates the effect of the feature set, not a
# simultaneously-varying K), compared via Adjusted Rand Index. This is a
# sanity check reported either way, not a threshold to pass — see README.
RFM_ONLY_FEATURES = ["frequency", "days_since_last_purchase", "total_calibration_spend"]


def load_inputs(processed_dir: Path, output_dir: Path) -> pd.DataFrame:
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)

    clv = pd.read_csv(output_dir / "clv_predictions.csv")[["customer_id", "predicted_clv"]]
    churn = pd.read_csv(output_dir / "churn_predictions.csv")[
        ["customer_id", "frequency", "days_since_last_purchase", "predicted_churn_proba"]
    ]
    invoices = pd.read_csv(processed_dir / "calibration_invoices.csv")
    total_spend = invoices.groupby("customer_id")["invoice_value"].sum().rename("total_calibration_spend")

    df = clv.merge(churn, on="customer_id", how="inner").merge(total_spend, on="customer_id", how="left")
    if len(df) != len(clv):
        raise ValueError(
            f"Merge lost/duplicated rows: clv={len(clv)}, churn={len(churn)}, merged={len(df)}. "
            "clv_model.py and churn_model.py should be running on the same calibration customer set."
        )
    return df


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, StandardScaler]:
    X = df[feature_cols].copy()
    for col in LOG_TRANSFORM_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def select_k(X: np.ndarray, k_candidates: list[int] = K_CANDIDATES) -> tuple[dict[int, float], dict[int, float], int]:
    """Elbow (inertia) and silhouette score for each candidate K.

    Silhouette is the deciding criterion (it's an actual scalar to
    maximize; the elbow point is often visually ambiguous on its own) —
    both are still plotted and reported together per the project spec.
    """
    inertias, silhouettes = {}, {}
    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
        inertias[k] = float(km.inertia_)
        silhouettes[k] = float(silhouette_score(X, km.labels_))
    best_k = max(silhouettes, key=silhouettes.get)
    return inertias, silhouettes, best_k


def plot_k_selection(inertias: dict[int, float], silhouettes: dict[int, float], best_k: int, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ks = sorted(inertias)

    ax1.plot(ks, [inertias[k] for k in ks], marker="o")
    ax1.axvline(best_k, color="crimson", linestyle="--", alpha=0.6)
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_title("Elbow method")

    ax2.plot(ks, [silhouettes[k] for k in ks], marker="o", color="darkorange")
    ax2.axvline(best_k, color="crimson", linestyle="--", alpha=0.6, label=f"chosen K={best_k}")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette method (deciding criterion)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def label_clusters(df: pd.DataFrame, cluster_col: str = "cluster") -> dict[int, str]:
    """Derive a human-readable label per cluster from its actual centroid values.

    Value tier compares each cluster's mean CLV against the *population
    median* — CLV has no inherent absolute reference point, so "high" only
    means "high relative to this customer base."

    Risk tier compares against the *fixed* classification threshold from
    churn_model.py (0.5 = "more likely than not to churn"), not the
    population median. Churn probability, unlike CLV, has a real absolute
    meaning — and here the churn classifier is so well-separated
    (ROC-AUC≈1.0, 54% base churn rate) that the population median sits at
    ~0.98, which would silently mislabel a cluster averaging 97% churn
    probability as "Low-Risk" simply for being a hair under that skewed
    median. Caught by inspecting the real segment_profiles.csv output,
    where a 97%-churn-probability cluster came out "Low-Risk" before this
    fix.

    Cluster ids are looked up, never assumed positional — KMeans cluster
    indices are arbitrary and change between runs/seeds.
    """
    clv_median = df["predicted_clv"].median()

    labels = {}
    for cluster_id, group in df.groupby(cluster_col):
        value_tier = "High-Value" if group["predicted_clv"].mean() >= clv_median else "Low-Value"
        risk_tier = "At-Risk" if group["predicted_churn_proba"].mean() >= CLASSIFICATION_THRESHOLD else "Low-Risk"
        labels[cluster_id] = f"{value_tier} / {risk_tier}"
    return labels


def compute_segment_profiles(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    profile = df.groupby(cluster_col).agg(
        label=("segment_label", "first"),
        n_customers=("customer_id", "count"),
        mean_predicted_clv=("predicted_clv", "mean"),
        mean_churn_proba=("predicted_churn_proba", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_days_since_last_purchase=("days_since_last_purchase", "mean"),
        mean_total_calibration_spend=("total_calibration_spend", "mean"),
    ).reset_index()
    profile["pct_of_customers"] = profile["n_customers"] / profile["n_customers"].sum()
    return profile.sort_values("mean_predicted_clv", ascending=False).reset_index(drop=True)


def plot_segment_scatter(df: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, group in df.groupby("segment_label"):
        ax.scatter(
            group["predicted_clv"].clip(lower=0), group["predicted_churn_proba"],
            alpha=0.4, s=12, label=f"{label} (n={len(group)})",
        )
    ax.set_xscale("symlog")
    ax.set_xlabel("Predicted 6-month CLV (£, symlog)")
    ax.set_ylabel("Predicted churn probability")
    ax.set_title("Customer segments: CLV vs. churn risk")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_rfm_only_stability(df: pd.DataFrame, full_labels: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    X_rfm, _ = prepare_feature_matrix(df, RFM_ONLY_FEATURES)
    km_rfm = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X_rfm)
    ari = float(adjusted_rand_score(full_labels, km_rfm.labels_))
    return km_rfm.labels_, ari


def run(processed_dir: Path, output_dir: Path, metrics_path: Path) -> None:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    models_dir = output_dir / "models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(processed_dir, output_dir)

    X, scaler = prepare_feature_matrix(df, FULL_FEATURES)
    inertias, silhouettes, best_k = select_k(X)
    plot_k_selection(inertias, silhouettes, best_k, plots_dir / "segmentation_k_selection.png")

    kmeans = KMeans(n_clusters=best_k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
    df["cluster"] = kmeans.labels_

    cluster_labels = label_clusters(df)
    df["segment_label"] = df["cluster"].map(cluster_labels)

    profiles = compute_segment_profiles(df)
    plot_segment_scatter(df, plots_dir / "segmentation_scatter.png")

    rfm_only_labels, ari = compute_rfm_only_stability(df, kmeans.labels_, best_k)

    joblib.dump(kmeans, models_dir / "kmeans.pkl")
    joblib.dump(scaler, models_dir / "segmentation_scaler.pkl")

    df[["customer_id", "cluster", "segment_label"] + FULL_FEATURES].to_csv(output_dir / "segments.csv", index=False)
    profiles.to_csv(output_dir / "segment_profiles.csv", index=False)

    log_metrics(
        stage="segmentation",
        metrics={
            "n_customers": int(len(df)),
            "k_candidates": K_CANDIDATES,
            "inertias": inertias,
            "silhouette_scores": silhouettes,
            "chosen_k": best_k,
            "cluster_labels": cluster_labels,
            "cluster_sizes": df["cluster"].value_counts().to_dict(),
            "rfm_only_stability_ari": ari,
        },
        path=metrics_path,
    )

    print(f"Chosen K={best_k} (silhouette={silhouettes[best_k]:.3f})")
    print("Cluster labels:", cluster_labels)
    print(f"RFM-only stability check: ARI={ari:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Means customer segmentation on CLV + churn + RFM.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics.json"))
    args = parser.parse_args()
    run(args.processed_dir, args.output_dir, args.metrics_path)


if __name__ == "__main__":
    main()
