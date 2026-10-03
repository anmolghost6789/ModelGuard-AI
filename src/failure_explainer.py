# src/failure_explainer.py
"""
MODULE 9 & 10: Failure Explanation & Explainable AI (XAI)
Decomposes Model Failure Risk Score into exact percentage contributions
across Data Quality, Distribution Drift, and Prediction Uncertainty,
generating actionable operator recommendations without unwarranted causal claims.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from src.failure_model import META_FEATURE_GROUPS


class FailureExplainer:
    """
    Explainability engine for model failure risk predictions.
    """
    def __init__(self, meta_model: Any, meta_features: List[str]):
        self.meta_model = meta_model
        self.meta_features = meta_features
        self.feature_importances = self._extract_importances()

    def _extract_importances(self) -> Dict[str, float]:
        """Extracts normalized feature importance weights from meta-model."""
        if hasattr(self.meta_model, "feature_importances_"):
            imps = self.meta_model.feature_importances_
        elif hasattr(self.meta_model, "coef_"):
            imps = np.abs(self.meta_model.coef_[0])
        else:
            imps = np.ones(len(self.meta_features)) / len(self.meta_features)

        imps = np.asarray(imps)
        norm_imps = imps / (imps.sum() + 1e-9)
        return {f: float(v) for f, v in zip(self.meta_features, norm_imps)}

    def decompose_risk_factors(
        self,
        telemetry_vector: Dict[str, float],
        risk_score: float
    ) -> Dict[str, Any]:
        """
        Calculates percentage contribution of each operational risk pillar:
        - Distribution Shift %
        - Prediction Uncertainty %
        - Data Quality & Missingness %
        """
        group_weights = {"drift": 0.0, "uncertainty": 0.0, "quality": 0.0}

        # Weight by feature importance multiplied by normalized deviation
        feature_impacts = {}
        for feat, imp in self.feature_importances.items():
            val = float(telemetry_vector.get(feat, 0.0))
            # Magnitude of telemetry impact
            impact = imp * max(0.01, min(2.0, val if val > 0 else 0.05))
            feature_impacts[feat] = impact

            if feat in META_FEATURE_GROUPS["drift"]:
                group_weights["drift"] += impact
            elif feat in META_FEATURE_GROUPS["uncertainty"]:
                group_weights["uncertainty"] += impact
            elif feat in META_FEATURE_GROUPS["quality"]:
                group_weights["quality"] += impact

        total_impact = sum(group_weights.values()) + 1e-9
        group_percentages = {
            "distribution_shift_pct": float(round((group_weights["drift"] / total_impact) * 100, 1)),
            "prediction_uncertainty_pct": float(round((group_weights["uncertainty"] / total_impact) * 100, 1)),
            "data_quality_pct": float(round((group_weights["quality"] / total_impact) * 100, 1))
        }

        # Top 5 most influential individual meta-features
        top_features = sorted(feature_impacts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_factors_list = []
        for rank, (f_name, f_imp) in enumerate(top_features, start=1):
            category = "Distribution Shift" if f_name in META_FEATURE_GROUPS["drift"] else ("Prediction Uncertainty" if f_name in META_FEATURE_GROUPS["uncertainty"] else "Data Quality")
            top_factors_list.append({
                "rank": rank,
                "feature": f_name,
                "category": category,
                "observed_value": float(telemetry_vector.get(f_name, 0.0)),
                "relative_influence": float(round((f_imp / total_impact) * 100, 1))
            })

        # Generate Actionable Operator Recommendations
        recommendations = []
        if risk_score >= 61.0:
            recommendations.append("CRITICAL: Model requires immediate operational quarantine; switch to safe fallback heuristic.")
            if group_percentages["distribution_shift_pct"] > 40.0:
                recommendations.append("Severe covariate shift detected: Retrain primary model using recent operational window data.")
            if group_percentages["data_quality_pct"] > 30.0:
                recommendations.append("Sensor data quality failure: Inspect upstream data ingestion pipelines for sensor hardware dropouts.")
            if group_percentages["prediction_uncertainty_pct"] > 35.0:
                recommendations.append("Extreme prediction ambiguity: Predictions should not be executed autonomously without human-in-the-loop review.")
        elif risk_score >= 31.0:
            recommendations.append("WARNING: Elevated operational risk. Increase monitoring frequency and log prediction errors.")
            if group_percentages["prediction_uncertainty_pct"] > 35.0:
                recommendations.append("Prediction confidence degraded: Route borderline predictions (confidence < 65%) to human verifiers.")
        else:
            recommendations.append("NORMAL: Primary model is operating within verified statistical tolerances. Safe for autonomous deployment.")

        return {
            "risk_score": float(risk_score),
            "group_percentages": group_percentages,
            "top_risk_factors": top_factors_list,
            "recommendations": recommendations,
            "disclaimer": "Attributions represent statistical associations with model degradation, not definitive causal claims."
        }
