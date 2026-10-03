# src/failure_predictor.py
"""
MODULE 8: Model Failure Risk Prediction Framework
Integrates Data Quality, Distribution Drift, and Prediction Uncertainty
into a secondary meta-model that predicts Model Failure Risk Score (0-100).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.drift_detector import DistributionShiftDetector
from src.uncertainty import compute_batch_uncertainty_summary
from src.failure_model import ALL_META_FEATURES


class ModelFailurePredictor:
    """
    Early-warning meta-system that predicts the probability of primary model failure
    without requiring ground-truth labels.
    """
    def __init__(
        self,
        meta_model: Optional[Any] = None,
        meta_scaler: Optional[Any] = None,
        risk_thresholds: Optional[Dict[str, float]] = None
    ):
        self.meta_model = meta_model
        self.meta_scaler = meta_scaler
        self.risk_thresholds = risk_thresholds or {
            "low_max": 30.0,
            "medium_max": 60.0
        }

    @classmethod
    def from_saved_artifacts(
        cls,
        model_path: str = "output/models/failure_rf_meta_model.joblib",
        scaler_path: str = "output/models/meta_scaler.joblib"
    ) -> "ModelFailurePredictor":
        """Loads serialized meta-model and scaler from disk."""
        meta_model = joblib.load(model_path) if os.path.exists(model_path) else None
        meta_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        return cls(meta_model=meta_model, meta_scaler=meta_scaler)

    def calculate_risk_level(self, risk_score: float) -> Tuple[str, str]:
        """
        Maps 0-100 failure risk score to operational alert levels.
        """
        if risk_score <= self.risk_thresholds["low_max"]:
            return "Low Risk", "NORMAL"
        elif risk_score <= self.risk_thresholds["medium_max"]:
            return "Medium Risk", "WARNING"
        else:
            return "High Risk", "HIGH RISK"

    def extract_telemetry_vector(
        self,
        X_batch: pd.DataFrame,
        primary_model: Any,
        ref_df: pd.DataFrame,
        scaler: Any,
        data_quality_info: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Extracts the 21-dimensional meta-feature telemetry vector from an operational batch.
        """
        # 1. Distribution Drift
        detector = DistributionShiftDetector(reference_df=ref_df)
        drift_stats = detector.compute_overall_drift(X_batch)

        # 2. Prediction Uncertainty
        X_sc = pd.DataFrame(scaler.transform(X_batch), columns=X_batch.columns)
        if hasattr(primary_model, "predict_proba"):
            probs = primary_model.predict_proba(X_sc)
        else:
            d = primary_model.predict(X_sc)
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
            probs = np.column_stack([1 - d_norm, d_norm])

        unc_stats = compute_batch_uncertainty_summary(probs, model=primary_model, X=X_sc)

        # 3. Data Quality
        dq = data_quality_info or {}
        miss_pct = dq.get("missing_pct", float(X_batch.isnull().mean().mean()))
        outlier_pct = dq.get("outlier_pct", 0.0)
        noise_std = dq.get("noise_std", 0.0)
        dq_score = dq.get("data_quality_score", max(0.0, 1.0 - (miss_pct * 1.5 + outlier_pct * 2.0)))

        vec = {
            "missing_pct": float(miss_pct),
            "noise_std": float(noise_std),
            "outlier_pct": float(outlier_pct),
            "cov_shift_mag": float(dq.get("cov_shift_mag", 0.0)),
            "corrupt_feature_cnt": float(dq.get("corrupt_feature_cnt", 0.0)),
            "data_quality_score": float(dq_score),

            "mean_ks_stat": float(drift_stats["mean_ks_stat"]),
            "max_ks_stat": float(drift_stats["max_ks_stat"]),
            "mean_psi": float(drift_stats["mean_psi"]),
            "max_psi": float(drift_stats["max_psi"]),
            "mean_js_divergence": float(drift_stats["mean_js_divergence"]),
            "pct_features_shifted": float(drift_stats["pct_features_shifted"]),
            "num_features_shifted": float(drift_stats["num_features_shifted"]),

            "mean_confidence": float(unc_stats["mean_confidence"]),
            "min_confidence": float(unc_stats["min_confidence"]),
            "mean_entropy": float(unc_stats["mean_entropy"]),
            "max_entropy": float(unc_stats["max_entropy"]),
            "pct_high_entropy": float(unc_stats["pct_high_entropy"]),
            "pct_low_confidence": float(unc_stats["pct_low_confidence"]),
            "mean_margin": float(unc_stats["mean_margin"]),
            "mean_ensemble_variance": float(unc_stats["mean_ensemble_variance"])
        }

        df_vec = pd.DataFrame([vec])[ALL_META_FEATURES]
        return df_vec

    def predict_failure_risk(
        self,
        telemetry_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculates failure risk probability and calibrated score (0-100).
        """
        if self.meta_model is None or self.meta_scaler is None:
            # Fallback heuristic if meta-model not yet loaded
            mean_psi = telemetry_df["mean_psi"].iloc[0]
            mean_ent = telemetry_df["mean_entropy"].iloc[0]
            prob = float(np.clip(mean_psi * 0.5 + mean_ent * 0.5, 0.0, 1.0))
        else:
            X_sc = self.meta_scaler.transform(telemetry_df)
            prob = float(self.meta_model.predict_proba(X_sc)[0, 1])

        risk_score = float(np.clip(prob * 100.0, 0.0, 100.0))
        risk_level, alert_status = self.calculate_risk_level(risk_score)

        return {
            "failure_probability": prob,
            "failure_risk_score": risk_score,
            "risk_level": risk_level,
            "alert_status": alert_status,
            "telemetry_features": telemetry_df.iloc[0].to_dict()
        }
