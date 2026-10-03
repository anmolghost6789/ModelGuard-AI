# src/failure_model.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

META_FEATURE_GROUPS = {
    "quality": [
        "missing_pct",
        "noise_std",
        "outlier_pct",
        "cov_shift_mag",
        "corrupt_feature_cnt",
        "data_quality_score"
    ],
    "drift": [
        "mean_ks_stat",
        "max_ks_stat",
        "mean_psi",
        "max_psi",
        "mean_js_divergence",
        "pct_features_shifted",
        "num_features_shifted"
    ],
    "uncertainty": [
        "mean_confidence",
        "min_confidence",
        "mean_entropy",
        "max_entropy",
        "pct_high_entropy",
        "pct_low_confidence",
        "mean_margin",
        "mean_ensemble_variance"
    ]
}

ALL_META_FEATURES = (
    META_FEATURE_GROUPS["quality"] +
    META_FEATURE_GROUPS["drift"] +
    META_FEATURE_GROUPS["uncertainty"]
)


def get_failure_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Returns candidate models for predicting model failure."""
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1
        )
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
    return models


def evaluate_failure_model_cv(
    model: Any,
    X_meta: pd.DataFrame,
    y_failure: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates failure model using Stratified K-Fold Cross Validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_meta)

    # Cross-validated out-of-fold probability predictions
    y_probs = cross_val_predict(
        model,
        X_scaled,
        y_failure,
        cv=skf,
        method="predict_proba"
    )[:, 1]

    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_failure, y_pred)
    prec = precision_score(y_failure, y_pred, zero_division=0)
    rec = recall_score(y_failure, y_pred, zero_division=0)
    f1 = f1_score(y_failure, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_failure, y_probs)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_failure, y_probs)
    except Exception:
        pr_auc = 0.5
        
    cm = confusion_matrix(y_failure, y_pred)
    brier = brier_score_loss(y_failure, y_probs)

    # Fit final model on all data
    final_model = model.__class__(**model.get_params())
    final_model.fit(X_scaled, y_failure)

    # Extract feature importances if tree-based, or coefficients if linear
    feature_imp = {}
    if hasattr(final_model, "feature_importances_"):
        for f, imp in zip(X_meta.columns, final_model.feature_importances_):
            feature_imp[f] = float(imp)
    elif hasattr(final_model, "coef_"):
        for f, coef in zip(X_meta.columns, final_model.coef_[0]):
            feature_imp[f] = float(abs(coef))

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "confusion_matrix": cm,
        "y_probs": y_probs,
        "y_pred": y_pred,
        "y_true": y_failure.values,
        "fitted_model": final_model,
        "scaler": scaler,
        "feature_importances": feature_imp
    }


def compare_feature_ablation(
    y_failure: pd.Series,
    df_meta: pd.DataFrame,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates ablation configurations to test Research Question 3:
    1. Quality-only features
    2. Drift-only features
    3. Uncertainty-only features
    4. Proposed Composite Model (Quality + Drift + Uncertainty)
    """
    ablation_results = {}
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state, n_jobs=-1)

    subsets = {
        "Quality Only": META_FEATURE_GROUPS["quality"],
        "Drift Only": META_FEATURE_GROUPS["drift"],
        "Uncertainty Only": META_FEATURE_GROUPS["uncertainty"],
        "Composite (Full System)": ALL_META_FEATURES
    }

    for name, cols in subsets.items():
        X_sub = df_meta[cols]
        res = evaluate_failure_model_cv(clf, X_sub, y_failure, random_state=random_state)
        ablation_results[name] = {
            "num_features": len(cols),
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "roc_auc": res["roc_auc"],
            "pr_auc": res["pr_auc"],
            "brier_score": res["brier_score"]
        }

    return ablation_results
