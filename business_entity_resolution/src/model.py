"""
model.py — Model training, cross-validation, and threshold optimization.

Compares: Logistic Regression, Decision Tree, Random Forest, HistGBT.
Uses GroupKFold to avoid source-1 entity leakage across folds.
Sweeps threshold from 0.05 to 0.95 to optimize F0.5.
Saves the best model + threshold + feature columns to artifacts/.
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import config
import features as F

config.ensure_dirs()


# ---------------------------------------------------------------------------
# F0.5 metric
# ---------------------------------------------------------------------------
def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """F_beta score. beta=0.5 weights precision 4x over recall."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def compute_f05_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    """Compute precision, recall, F0.5 at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05       = f_beta(precision, recall, beta=0.5)
    return precision, recall, f05


def sweep_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """
    Sweep threshold from SWEEP_MIN to SWEEP_MAX.
    Returns (best_threshold, best_f05, precision_at_best, recall_at_best).
    """
    thresholds = np.arange(
        config.THRESHOLD_SWEEP_MIN,
        config.THRESHOLD_SWEEP_MAX + 1e-9,
        config.THRESHOLD_SWEEP_STEP,
    )
    best_t, best_f05, best_p, best_r = 0.5, 0.0, 0.0, 0.0
    for t in thresholds:
        p, r, f = compute_f05_at_threshold(y_true, y_score, t)
        if f > best_f05:
            best_t, best_f05, best_p, best_r = t, f, p, r
    return best_t, best_f05, best_p, best_r


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def get_models():
    """Return dict of model_name -> sklearn estimator (or Pipeline)."""
    rs = config.RANDOM_STATE
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=500, solver="lbfgs", random_state=rs
            )),
        ]),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=rs
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=4, random_state=rs
        ),
        "HistGBT": HistGradientBoostingClassifier(
            max_iter=300, max_depth=8, learning_rate=0.05,
            min_samples_leaf=20, random_state=rs
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation with GroupKFold
# ---------------------------------------------------------------------------
def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = config.N_CV_FOLDS,
) -> dict:
    """
    GroupKFold CV. Groups = source1_entity_id (prevents S1 leakage).
    Returns dict with OOF scores, best threshold, per-fold metrics.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_scores = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        oof_scores[val_idx] = model.predict_proba(X_val)[:, 1]

        t, f05, p, r = sweep_threshold(y_val, oof_scores[val_idx])
        fold_metrics.append({"fold": fold+1, "threshold": t, "f05": f05, "precision": p, "recall": r})

    # Global OOF threshold sweep
    best_t, best_f05, best_p, best_r = sweep_threshold(y, oof_scores)

    return {
        "oof_scores":      oof_scores,
        "best_threshold":  best_t,
        "oof_f05":         best_f05,
        "oof_precision":   best_p,
        "oof_recall":      best_r,
        "fold_metrics":    fold_metrics,
    }


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------
def train_and_select(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list,
    report_path: str = None,
) -> dict:
    """
    Train all models, select best by OOF F0.5, retrain on full data.
    Returns dict with model, threshold, feature_names, model_name, metrics.
    """
    models = get_models()
    results = {}

    print("\n  Model cross-validation (GroupKFold):")
    print(f"  {'Model':25s} {'OOF F0.5':>10s} {'Precision':>10s} {'Recall':>10s} {'Threshold':>10s}")
    print("  " + "-" * 70)

    for name, model in models.items():
        t0 = time.time()
        cv = cross_validate_model(model, X, y, groups)
        elapsed = time.time() - t0
        results[name] = cv
        results[name]["model"] = model
        print(
            f"  {name:25s} {cv['oof_f05']:>10.4f} "
            f"{cv['oof_precision']:>10.4f} {cv['oof_recall']:>10.4f} "
            f"{cv['best_threshold']:>10.3f}  ({elapsed:.0f}s)"
        )

    # Select best model by OOF F0.5 (prefer simpler if within fold variation)
    best_name = max(results, key=lambda k: results[k]["oof_f05"])
    best_result = results[best_name]
    print(f"\n  Selected: {best_name} (OOF F0.5 = {best_result['oof_f05']:.4f})")

    # Retrain on full data with best model
    best_model = get_models()[best_name]
    best_model.fit(X, y)

    artifact = {
        "model":          best_model,
        "threshold":      best_result["best_threshold"],
        "feature_names":  feature_names,
        "model_name":     best_name,
        "oof_f05":        best_result["oof_f05"],
        "oof_precision":  best_result["oof_precision"],
        "oof_recall":     best_result["oof_recall"],
        "all_results":    {k: {kk: vv for kk, vv in v.items() if kk != "model"
                               and kk != "oof_scores"}
                           for k, v in results.items()},
    }

    # Save model
    joblib.dump(artifact, config.MODEL_PATH)
    print(f"  Model saved to: {config.MODEL_PATH}")

    # Save feature columns
    with open(config.FEATURE_COLS_PATH, "w") as f:
        json.dump(feature_names, f)

    if report_path:
        _write_model_report(artifact, results, report_path)

    return artifact


def _write_model_report(artifact: dict, results: dict, path: str):
    lines = [
        "# Model Training Report\n",
        f"Selected model: **{artifact['model_name']}**\n",
        f"OOF F0.5: {artifact['oof_f05']:.4f}\n",
        f"Precision: {artifact['oof_precision']:.4f}\n",
        f"Recall: {artifact['oof_recall']:.4f}\n",
        f"Threshold: {artifact['threshold']:.3f}\n\n",
        "## All Model Results\n",
        "| Model | OOF F0.5 | Precision | Recall | Threshold |\n",
        "|-------|----------|-----------|--------|----------|\n",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['oof_f05']:.4f} | {r['oof_precision']:.4f} "
            f"| {r['oof_recall']:.4f} | {r['best_threshold']:.3f} |\n"
        )
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  Model report: {path}")


def load_model() -> dict:
    """Load saved model artifact."""
    return joblib.load(config.MODEL_PATH)
