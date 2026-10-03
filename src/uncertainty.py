# src/uncertainty.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


def compute_instance_uncertainty(
    probs: np.ndarray,
    model: Optional[Any] = None,
    X: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes per-instance uncertainty metrics:
    - confidence: max class probability in [0.5, 1.0]
    - entropy: normalized Shannon entropy in [0.0, 1.0]
    - margin: difference between top two probabilities
    - ensemble_variance: variance across RF trees if model is RandomForest
    """
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    
    # Binary classification probabilities
    if probs.ndim == 1 or probs.shape[1] == 1:
        p1 = probs.ravel()
        p0 = 1.0 - p1
        probs_2d = np.column_stack([p0, p1])
    else:
        probs_2d = probs

    # Confidence (max probability)
    conf = np.max(probs_2d, axis=1)

    # Normalized Shannon Entropy: - sum(p * ln(p)) / ln(C)
    num_classes = probs_2d.shape[1]
    norm_factor = np.log(num_classes) if num_classes > 1 else 1.0
    entropy = -np.sum(probs_2d * np.log(probs_2d), axis=1) / norm_factor
    entropy = np.clip(entropy, 0.0, 1.0)

    # Margin: top probability minus second top probability
    sorted_probs = np.sort(probs_2d, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    df_res = pd.DataFrame({
        "confidence": conf,
        "entropy": entropy,
        "margin": margin
    })

    # If Random Forest, compute ensemble disagreement (variance across trees)
    if model is not None and hasattr(model, "estimators_") and X is not None:
        try:
            tree_preds = np.array([tree.predict_proba(X.values)[:, 1] for tree in model.estimators_])
            tree_var = np.var(tree_preds, axis=0)
            df_res["ensemble_variance"] = tree_var
        except Exception:
            df_res["ensemble_variance"] = 0.0
    else:
        df_res["ensemble_variance"] = 0.0

    return df_res


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, pd.DataFrame]:
    """
    Computes Expected Calibration Error (ECE) and binning statistics.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_data = []
    total_samples = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        mask = (bin_indices == i)
        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(y_prob[mask]))
            avg_acc = float(np.mean(y_true[mask]))
            weight = count / total_samples
            ece += weight * abs(avg_acc - avg_conf)
            bin_data.append({
                "bin": i,
                "bin_center": float((bin_edges[i] + bin_edges[i+1]) / 2),
                "count": count,
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "gap": abs(avg_acc - avg_conf)
            })
        else:
            bin_data.append({
                "bin": i,
                "bin_center": float((bin_edges[i] + bin_edges[i+1]) / 2),
                "count": 0,
                "avg_confidence": float((bin_edges[i] + bin_edges[i+1]) / 2),
                "avg_accuracy": 0.0,
                "gap": 0.0
            })

    return float(ece), pd.DataFrame(bin_data)


def compute_batch_uncertainty_summary(
    probs: np.ndarray,
    model: Optional[Any] = None,
    X: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Aggregates instance uncertainty into batch-level summary features.
    """
    df_unc = compute_instance_uncertainty(probs, model=model, X=X)
    
    return {
        "mean_confidence": float(df_unc["confidence"].mean()),
        "min_confidence": float(df_unc["confidence"].min()),
        "mean_entropy": float(df_unc["entropy"].mean()),
        "max_entropy": float(df_unc["entropy"].max()),
        "pct_high_entropy": float((df_unc["entropy"] > 0.70).mean()),
        "pct_low_confidence": float((df_unc["confidence"] < 0.65).mean()),
        "mean_margin": float(df_unc["margin"].mean()),
        "mean_ensemble_variance": float(df_unc["ensemble_variance"].mean())
    }


def audit_prediction_samples(
    probs: np.ndarray,
    class_labels: Optional[List[str]] = None,
    threshold: float = 0.5,
    high_entropy_cutoff: float = 0.70,
    low_confidence_cutoff: float = 0.65
) -> pd.DataFrame:
    """
    Audits prediction samples, assigning confidence, uncertainty grade,
    and actionable operator review tags (e.g. 'Prediction should be reviewed').
    """
    labels = class_labels or ["Negative", "Positive"]
    df_unc = compute_instance_uncertainty(probs)

    if probs.ndim == 1 or probs.shape[1] == 1:
        p1 = probs.ravel()
    else:
        p1 = probs[:, 1]

    preds = (p1 >= threshold).astype(int)
    pred_names = [labels[p] if p < len(labels) else f"Class {p}" for p in preds]

    audit_rows = []
    for i in range(len(preds)):
        conf = df_unc["confidence"].iloc[i]
        ent = df_unc["entropy"].iloc[i]

        if ent >= high_entropy_cutoff or conf <= low_confidence_cutoff:
            unc_level = "High"
            action = "Prediction should be reviewed"
        elif ent >= 0.50 or conf <= 0.80:
            unc_level = "Moderate"
            action = "Monitor closely"
        else:
            unc_level = "Low"
            action = "Approved"

        audit_rows.append({
            "Sample Index": i,
            "Prediction": pred_names[i],
            "Confidence": f"{conf * 100:.1f}%",
            "Confidence Value": float(conf),
            "Shannon Entropy": float(ent),
            "Uncertainty Level": unc_level,
            "Action Recommendation": action
        })

    return pd.DataFrame(audit_rows)
