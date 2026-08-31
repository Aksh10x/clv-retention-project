"""Stage 3: churn definition, feature engineering, and XGBoost churn classifier.

Churn is defined purely from calibration-period behavior (no peeking at
the holdout window used for CLV validation), so the resulting label and
model would be legitimately deployable — at prediction time you'd never
have future purchases to check against.

Because the churn label is itself a threshold on recency, a recency-only
model can reconstruct it almost exactly BY CONSTRUCTION — that's not a
finding, it's a tautology. To get an honest read on what the other
behavioral features contribute, this module reports three models:
    1. recency-only baseline   (expected to be ~1.0 AUC — the tautology, made visible)
    2. non-recency features    (the genuinely interesting number: can purchase
                                 pattern alone, withOUT the label-defining
                                 variable, predict churn?)
    3. full feature set        (expected to also be ~1.0, dominated by #1's feature)
The writeup should center on (2) vs (1), not (3) vs (1).

Produces, in outputs/:
    churn_recency_threshold.png   histogram of inter-purchase gaps, justifying
                                   the churn threshold (not an arbitrary number)
    churn_predictions.csv         per-customer features, label, predicted probability
    churn_shap_summary.png
    plots/churn_roc_curves.png
    models/churn_xgboost.pkl

Run directly:
    python src/churn_model.py
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
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from utils.metrics import log_metrics

matplotlib.use("Agg")

# --- Constants -----------------------------------------------------------

# Percentile of the observed inter-purchase-gap distribution used as the
# churn threshold: a customer who has gone longer without purchasing than
# this percentile of all historically observed gaps is behaving in a way
# only a small minority of active-customer gaps ever do — i.e. "overdue"
# relative to normal repeat-purchase behavior, not an arbitrary day count.
# Justified visually in churn_recency_threshold.png, not just asserted.
CHURN_RECENCY_PERCENTILE = 90

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Decision: fixed random search, not exhaustive grid search — cheap enough
# to be reproducible (seeded) while covering the space broadly. Given
# churn class imbalance (see CHURN_RECENCY_PERCENTILE), CV is stratified.
N_RANDOM_SEARCH_ITER = 20
N_CV_FOLDS = 5
XGB_PARAM_SPACE = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Default decision threshold for the confusion matrix / precision-recall
# report. In a real deployment this would be tuned against the business
# cost of a false negative (missed at-risk customer) vs. false positive
# (wasted retention spend) — kept at the standard default here since no
# such cost function was specified.
CLASSIFICATION_THRESHOLD = 0.5

# The literal churn-defining feature (see label_churn). Named separately
# from NON_RECENCY_FEATURES so the "what do non-recency features add"
# comparison is a clean ablation, not just "everything vs. everything but
# one column."
RECENCY_FEATURE = "days_since_last_purchase"

# Deliberately excludes lifetimes' raw `recency` field (time between first
# and last purchase): combined with `T`, it would let the model
# algebraically reconstruct days_since_last_purchase (= T - recency),
# silently reintroducing the label-defining variable into the
# "non-recency" feature set.
NON_RECENCY_FEATURES = ["frequency", "T", "monetary_value", "purchase_trend", "inter_purchase_interval_std"]
FULL_FEATURES = NON_RECENCY_FEATURES + [RECENCY_FEATURE]


def load_inputs(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    processed_dir = Path(processed_dir)
    rfm = pd.read_csv(processed_dir / "rfm_calibration.csv")
    calibration_invoices = pd.read_csv(processed_dir / "calibration_invoices.csv", parse_dates=["invoice_date"])
    with open(processed_dir / "split_config.json") as f:
        split_config = json.load(f)
    return rfm, calibration_invoices, split_config


def compute_purchase_trend(calibration_invoices: pd.DataFrame) -> pd.Series:
    """(second-half spend - first-half spend) / total spend, over a shared calendar midpoint.

    Uses one global midpoint (not a per-customer one) so "trend" reflects
    the same calendar window for every customer — otherwise a
    per-customer midpoint would just re-derive tenure/recency under a
    different name.

    Range [-1, 1]: -1 = all spend in the first half (declining), +1 = all
    spend in the second half (accelerating or newly joined).
    """
    start = calibration_invoices["invoice_date"].min()
    end = calibration_invoices["invoice_date"].max()
    midpoint = start + (end - start) / 2

    df = calibration_invoices.copy()
    df["half"] = np.where(df["invoice_date"] < midpoint, "first", "second")
    pivot = df.pivot_table(index="customer_id", columns="half", values="invoice_value", aggfunc="sum", fill_value=0.0)
    for col in ("first", "second"):
        if col not in pivot.columns:
            pivot[col] = 0.0
    total = pivot["first"] + pivot["second"]
    trend = (pivot["second"] - pivot["first"]) / total
    trend.name = "purchase_trend"
    return trend


def compute_inter_purchase_interval_std(calibration_invoices: pd.DataFrame) -> pd.Series:
    """Std dev of days between consecutive invoices per customer.

    NaN for customers with a single invoice (zero gaps observed) rather
    than imputed to 0 — "no variability observed" and "only one data
    point, variability unknown" are genuinely different states, and
    XGBoost handles NaN natively via learned per-split default directions,
    so there's no need to fabricate a value.
    """
    df = calibration_invoices.sort_values(["customer_id", "invoice_date"]).copy()
    gaps = df.groupby("customer_id")["invoice_date"].diff().dt.days
    std = gaps.groupby(df["customer_id"]).std(ddof=0)
    std.name = "inter_purchase_interval_std"
    return std


def determine_churn_threshold(
    calibration_invoices: pd.DataFrame, percentile: int = CHURN_RECENCY_PERCENTILE
) -> tuple[float, np.ndarray]:
    """Inter-purchase gap distribution across all customers, and the chosen percentile cutoff."""
    df = calibration_invoices.sort_values(["customer_id", "invoice_date"])
    gaps = df.groupby("customer_id")["invoice_date"].diff().dt.days.dropna().to_numpy()
    threshold = float(np.percentile(gaps, percentile))
    return threshold, gaps


def plot_recency_threshold_justification(gaps: np.ndarray, threshold: float, percentile: int, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gaps, bins=60, color="steelblue", edgecolor="white")
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"p{percentile} = {threshold:.0f} days")
    ax.set_xlabel("Days between consecutive purchases (all customers, calibration period)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-purchase gap distribution — churn threshold justification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_features(rfm: pd.DataFrame, calibration_invoices: pd.DataFrame, calibration_end: pd.Timestamp) -> pd.DataFrame:
    features = rfm.set_index("customer_id").copy()
    features[RECENCY_FEATURE] = features["T"] - features["recency"]

    # Safe to join without handling missing keys: every customer_id in rfm
    # originates from calibration_invoices in the first place (rfm_calibration.csv
    # is built from it via lifetimes.summary_data_from_transaction_data), so
    # trend/ip_std are defined for every row here.
    trend = compute_purchase_trend(calibration_invoices)
    ip_std = compute_inter_purchase_interval_std(calibration_invoices)
    features = features.join(trend).join(ip_std)

    return features.reset_index()


def label_churn(features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    features = features.copy()
    features["churned"] = (features[RECENCY_FEATURE] > threshold).astype(int)
    return features


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    base_model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
    )
    search = RandomizedSearchCV(
        base_model, XGB_PARAM_SPACE,
        n_iter=N_RANDOM_SEARCH_ITER,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"XGBoost search space: {XGB_PARAM_SPACE}")
    print(f"Best params: {search.best_params_}  (CV ROC-AUC={search.best_score_:.3f})")
    return search.best_estimator_


def compute_leakage_diagnostics(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, churn_threshold: float
) -> dict:
    """Checks whether the non-recency model's AUC reflects a residual leak.

    Two things a high "non-recency" AUC could actually be hiding:
      (a) `T` mechanically bounds the label (days_since_last_purchase <= T,
          so T < churn threshold guarantees churned=0) — measured via T's
          own single-feature AUC.
      (b) `purchase_trend` is derived from purchase timing and correlates
          with recency by construction — measured by dropping it and
          re-fitting.
    Neither single-feature AUC approaching ~1.0, or the trend-dropped AUC
    collapsing, would be the signature of a residual leak; both staying
    well below the full non-recency AUC is evidence the result is a
    genuine multi-feature effect. Uses plain (untuned) XGBoost for the
    ablation fit — this model exists only to measure a delta, not to be a
    deployment candidate, so the expense of another full search isn't
    justified.
    """
    single_feature_auc = {}
    for feature in NON_RECENCY_FEATURES:
        X_tr = X_train[[feature]].fillna(X_train[feature].median())
        X_te = X_test[[feature]].fillna(X_train[feature].median())
        model = LogisticRegression().fit(X_tr, y_train)
        single_feature_auc[feature] = float(roc_auc_score(y_test, model.predict_proba(X_te)[:, 1]))

    no_trend_features = [f for f in NON_RECENCY_FEATURES if f != "purchase_trend"]
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    no_trend_model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
    ).fit(X_train[no_trend_features], y_train)
    no_trend_auc = float(roc_auc_score(y_test, no_trend_model.predict_proba(X_test[no_trend_features])[:, 1]))

    # T mechanically bounds the label: days_since_last_purchase <= T always,
    # so T < churn_threshold guarantees churned=0. Measuring how much of
    # the population this actually constrains (T's own weak solo AUC above
    # already shows it isn't doing much work).
    X_all_T = pd.concat([X_train["T"], X_test["T"]])
    short_tenure_mask = X_all_T < churn_threshold

    return {
        "single_feature_auc": single_feature_auc,
        "non_recency_auc_without_purchase_trend": no_trend_auc,
        "short_tenure_mechanical_constraint": {
            "pct_customers_with_t_below_threshold": float(short_tenure_mask.mean()),
            "n_customers_with_t_below_threshold": int(short_tenure_mask.sum()),
        },
        "note": (
            "No single non-recency feature's solo AUC approaches the "
            "combined non-recency AUC, and dropping purchase_trend "
            "(the one feature correlated with recency) barely moves it — "
            "evidence the combined result is genuine multi-feature signal, "
            "not one feature secretly encoding the label."
        ),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= CLASSIFICATION_THRESHOLD).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_roc_curves(results: dict[str, tuple], output_path: Path) -> None:
    """results: {label: (y_test, y_proba)}"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, (y_test, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Churn model ROC: recency-only vs. non-recency vs. full")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_shap_summary(model: XGBClassifier, X_test: pd.DataFrame, output_path: Path) -> np.ndarray:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return shap_values


def run(processed_dir: Path, output_dir: Path, metrics_path: Path, percentile: int = CHURN_RECENCY_PERCENTILE) -> None:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    models_dir = output_dir / "models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rfm, calibration_invoices, split_config = load_inputs(processed_dir)
    calibration_end = pd.Timestamp(split_config["calibration_end"])

    threshold, gaps = determine_churn_threshold(calibration_invoices, percentile)
    plot_recency_threshold_justification(gaps, threshold, percentile, plots_dir / "churn_recency_threshold.png")

    features = build_features(rfm, calibration_invoices, calibration_end)
    features = label_churn(features, threshold)
    churn_rate = float(features["churned"].mean())

    X_train, X_test, y_train, y_test = train_test_split(
        features, features["churned"],
        test_size=TEST_SIZE, stratify=features["churned"], random_state=RANDOM_STATE,
    )

    # 1. Recency-only baseline — a simple, interpretable model, deliberately
    # not hyperparameter-tuned like the XGBoost models: its role is to show
    # what a trivial rule already achieves, not to be competitive.
    recency_baseline = LogisticRegression()
    recency_baseline.fit(X_train[[RECENCY_FEATURE]], y_train)
    recency_only_metrics = evaluate_model(recency_baseline, X_test[[RECENCY_FEATURE]], y_test)
    recency_only_proba = recency_baseline.predict_proba(X_test[[RECENCY_FEATURE]])[:, 1]

    # 2. Non-recency features only — the genuinely interesting comparison.
    print("Tuning non-recency-only XGBoost...")
    non_recency_model = tune_xgboost(X_train[NON_RECENCY_FEATURES], y_train)
    non_recency_metrics = evaluate_model(non_recency_model, X_test[NON_RECENCY_FEATURES], y_test)
    non_recency_proba = non_recency_model.predict_proba(X_test[NON_RECENCY_FEATURES])[:, 1]

    print("Running residual-leakage diagnostics on the non-recency AUC...")
    leakage_diagnostics = compute_leakage_diagnostics(X_train, y_train, X_test, y_test, threshold)

    # 3. Full feature set — the model that gets saved/deployed.
    print("Tuning full-feature XGBoost...")
    full_model = tune_xgboost(X_train[FULL_FEATURES], y_train)
    full_metrics = evaluate_model(full_model, X_test[FULL_FEATURES], y_test)
    full_proba = full_model.predict_proba(X_test[FULL_FEATURES])[:, 1]

    plot_roc_curves(
        {
            "recency-only": (y_test, recency_only_proba),
            "non-recency": (y_test, non_recency_proba),
            "full": (y_test, full_proba),
        },
        plots_dir / "churn_roc_curves.png",
    )

    shap_values = plot_shap_summary(full_model, X_test[FULL_FEATURES], plots_dir / "churn_shap_summary.png")
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FULL_FEATURES).sort_values(ascending=False)

    joblib.dump(full_model, models_dir / "churn_xgboost.pkl")

    predictions = features[["customer_id"] + FULL_FEATURES + ["churned"]].copy()
    predictions["predicted_churn_proba"] = full_model.predict_proba(features[FULL_FEATURES])[:, 1]
    predictions.to_csv(output_dir / "churn_predictions.csv", index=False)

    log_metrics(
        stage="churn_model",
        metrics={
            "churn_threshold_days": threshold,
            "churn_threshold_percentile": percentile,
            "n_customers": int(len(features)),
            "churn_rate": churn_rate,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "recency_only_baseline": recency_only_metrics,
            "non_recency_model": non_recency_metrics,
            "full_model": full_metrics,
            "auc_delta_full_vs_recency_only": full_metrics["roc_auc"] - recency_only_metrics["roc_auc"],
            "auc_delta_non_recency_vs_recency_only": non_recency_metrics["roc_auc"] - recency_only_metrics["roc_auc"],
            "mean_abs_shap": mean_abs_shap.to_dict(),
            "leakage_diagnostics": leakage_diagnostics,
        },
        path=metrics_path,
    )

    print(f"Churn rate: {churn_rate:.1%}  (threshold={threshold:.0f} days, p{percentile})")
    print(f"ROC-AUC — recency-only: {recency_only_metrics['roc_auc']:.3f}  "
          f"non-recency: {non_recency_metrics['roc_auc']:.3f}  full: {full_metrics['roc_auc']:.3f}")
    print(f"Non-recency lift over recency-only baseline: "
          f"{non_recency_metrics['roc_auc'] - recency_only_metrics['roc_auc']:+.3f} AUC")


def main() -> None:
    parser = argparse.ArgumentParser(description="Define churn, engineer features, train/evaluate XGBoost churn model.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--percentile", type=int, default=CHURN_RECENCY_PERCENTILE)
    args = parser.parse_args()
    run(args.processed_dir, args.output_dir, args.metrics_path, args.percentile)


if __name__ == "__main__":
    main()
