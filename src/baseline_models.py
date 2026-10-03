# src/baseline_models.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def get_baseline_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Initializes standard baseline models."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state,
            solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
    }
    
    # Try importing XGBoost if available
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
    except Exception:
        pass
        
    return models


def evaluate_classifier(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates classifier and returns comprehensive metrics.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        y_prob = probs[:, 1]
    else:
        y_prob = model.decision_function(X)
        # min-max normalize decision function to [0, 1] if needed
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
        
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y, y_prob)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y, y_prob)
    except Exception:
        pr_auc = 0.5
        
    cm = confusion_matrix(y, y_pred)
    brier = brier_score_loss(y, y_prob)
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "confusion_matrix": cm,
        "y_prob": y_prob,
        "y_pred": y_pred
    }


def evaluate_regression(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Evaluates continuous regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }
