# src/baseline_evaluator.py
"""
MODULE 4: Baseline Model Evaluation Engine
Comprehensive metric benchmarking for classification (Accuracy, Precision, Recall,
F1, ROC-AUC, PR-AUC, Confusion Matrix, Brier) and regression (MAE, MSE, RMSE, R2, MAPE).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

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
    r2_score,
    median_absolute_error
)


class BaselineEvaluator:
    """
    Standardized benchmarking engine for baseline machine learning models.
    """
    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type

    def evaluate_classification(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Calculates comprehensive classification performance metrics."""
        y_true = np.asarray(y_true)
        
        # Handle 1D or 2D probability outputs
        if y_prob.ndim == 2:
            if y_prob.shape[1] == 2:
                prob_positive = y_prob[:, 1]
            else:
                prob_positive = y_prob
        else:
            prob_positive = y_prob

        y_pred = (prob_positive >= threshold).astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            roc_auc = float(roc_auc_score(y_true, prob_positive))
        except Exception:
            roc_auc = 0.5

        try:
            pr_auc = float(average_precision_score(y_true, prob_positive))
        except Exception:
            pr_auc = 0.5

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
        brier = float(brier_score_loss(y_true, prob_positive)) if y_prob.ndim <= 2 else 0.0

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "confusion_matrix": cm,
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
            "y_prob": prob_positive,
            "y_pred": y_pred
        }

    def evaluate_regression(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates comprehensive regression performance metrics."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        medae = float(median_absolute_error(y_true, y_pred))

        # Mean Absolute Percentage Error (MAPE) handling zeros
        non_zero = y_true != 0
        if np.any(non_zero):
            mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
        else:
            mape = 0.0

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "medae": medae,
            "mape": mape,
            "residuals": y_true - y_pred,
            "y_pred": y_pred
        }

    def evaluate_model(
        self,
        model: Any,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Infers predictions and computes task-specific benchmark metrics."""
        if self.task_type == "classification":
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_eval)
            elif hasattr(model, "decision_function"):
                d = model.decision_function(X_eval)
                d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
                probs = np.column_stack([1 - d_norm, d_norm])
            else:
                preds = model.predict(X_eval)
                probs = np.column_stack([1 - preds, preds])
            return self.evaluate_classification(y_eval, probs, threshold=threshold)
        else:
            preds = model.predict(X_eval)
            return self.evaluate_regression(y_eval, preds)

    def benchmark_model_zoo(
        self,
        models_dict: Dict[str, Any],
        X_eval: pd.DataFrame,
        y_eval: pd.Series
    ) -> pd.DataFrame:
        """Evaluates an entire suite of models and outputs a comparative table."""
        records = []
        for name, m in models_dict.items():
            res = self.evaluate_model(m, X_eval, y_eval)
            if self.task_type == "classification":
                records.append({
                    "Model": name,
                    "Accuracy": res["accuracy"],
                    "Precision": res["precision"],
                    "Recall": res["recall"],
                    "F1-Score": res["f1"],
                    "ROC-AUC": res["roc_auc"],
                    "PR-AUC": res["pr_auc"],
                    "Brier Score": res["brier_score"]
                })
            else:
                records.append({
                    "Model": name,
                    "MAE": res["mae"],
                    "MSE": res["mse"],
                    "RMSE": res["rmse"],
                    "R2": res["r2"],
                    "MAPE %": res["mape"]
                })
        return pd.DataFrame(records)
