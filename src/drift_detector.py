# src/drift_detector.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

NUMERICAL_COLS = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL_COLS = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Computes Population Stability Index (PSI) using quantile binning defined on expected (train).
    """
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()
    
    # Handle constants or insufficient values
    if len(np.unique(expected)) <= 1 or len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Define bins based on expected quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    try:
        bin_edges = np.percentile(expected, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return 0.0
    except Exception:
        return 0.0

    # Count occurrences in each bin
    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Smooth with epsilon to avoid division by zero or log(0)
    exp_pct = np.clip(exp_pct, epsilon, 1.0)
    act_pct = np.clip(act_pct, epsilon, 1.0)

    # Normalize back to 1.0
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, psi_val))


def calculate_js_divergence(
    p_series: pd.Series,
    q_series: pd.Series
) -> float:
    """
    Computes Jensen-Shannon divergence between two categorical distributions.
    Bounded in [0, 1].
    """
    all_categories = sorted(list(set(p_series.unique()).union(set(q_series.unique()))))
    if len(all_categories) == 0:
        return 0.0
        
    p_counts = p_series.value_counts(normalize=True).reindex(all_categories, fill_value=0.0).values
    q_counts = q_series.value_counts(normalize=True).reindex(all_categories, fill_value=0.0).values

    # Add small epsilon
    eps = 1e-6
    p_counts = (p_counts + eps) / (p_counts + eps).sum()
    q_counts = (q_counts + eps) / (q_counts + eps).sum()

    # jensenshannon returns the distance (square root of divergence); square it to get divergence
    js_dist = jensenshannon(p_counts, q_counts, base=2)
    return float(js_dist ** 2)


class DistributionShiftDetector:
    """
    Measures distribution drift between training reference data
    and new/evaluation data.
    """
    def __init__(
        self,
        reference_df: pd.DataFrame,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ):
        self.ref_df = reference_df.copy()
        self.numerical_cols = numerical_cols or [c for c in NUMERICAL_COLS if c in reference_df.columns]
        self.categorical_cols = categorical_cols or [c for c in CATEGORICAL_COLS if c in reference_df.columns]

    def compute_feature_drift(
        self,
        eval_df: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculates drift metrics for every feature.
        """
        drift_results = {}

        # Continuous features
        for col in self.numerical_cols:
            if col not in eval_df.columns:
                continue
            ref_vals = self.ref_df[col].dropna().values
            eval_vals = eval_df[col].dropna().values

            ks_res = ks_2samp(ref_vals, eval_vals)
            psi_val = calculate_psi(ref_vals, eval_vals)

            severity = "Low"
            if psi_val >= 0.25 or ks_res.statistic >= 0.30:
                severity = "High"
            elif psi_val >= 0.10 or ks_res.statistic >= 0.15:
                severity = "Medium"

            drift_results[col] = {
                "type": "numerical",
                "ks_stat": float(ks_res.statistic),
                "ks_pvalue": float(ks_res.pvalue),
                "psi": float(psi_val),
                "primary_score": float(psi_val),
                "severity": severity,
                "risk_level": severity,
                "drift_detected": bool(ks_res.pvalue < alpha or psi_val >= 0.10)
            }

        # Categorical features
        for col in self.categorical_cols:
            if col not in eval_df.columns:
                continue
            ref_s = self.ref_df[col].dropna()
            eval_s = eval_df[col].dropna()

            jsd = calculate_js_divergence(ref_s, eval_s)
            ref_vc = ref_s.value_counts(normalize=True).to_dict()
            eval_vc = eval_s.value_counts(normalize=True).to_dict()
            all_cats = set(ref_vc.keys()).union(set(eval_vc.keys()))
            
            eps = 1e-4
            cat_psi = 0.0
            for cat in all_cats:
                p_c = ref_vc.get(cat, eps)
                q_c = eval_vc.get(cat, eps)
                cat_psi += (q_c - p_c) * np.log(q_c / p_c)
            cat_psi = float(max(0.0, cat_psi))

            severity = "Low"
            if cat_psi >= 0.25 or jsd >= 0.15:
                severity = "High"
            elif cat_psi >= 0.10 or jsd >= 0.05:
                severity = "Medium"

            drift_results[col] = {
                "type": "categorical",
                "js_divergence": float(jsd),
                "psi": float(cat_psi),
                "primary_score": float(cat_psi),
                "severity": severity,
                "risk_level": severity,
                "drift_detected": bool(jsd > 0.05 or cat_psi >= 0.10)
            }

        return drift_results

    def get_feature_drift_table(self, eval_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        """
        Returns clean summary table of feature drift matching Module 5 specification:
        Feature | Shift | Risk
        """
        drift_dict = self.compute_feature_drift(eval_df, alpha=alpha)
        rows = []
        for feat, d in drift_dict.items():
            metric_name = "PSI & KS" if d["type"] == "numerical" else "PSI & JSD"
            rows.append({
                "Feature": feat,
                "Type": d["type"].capitalize(),
                "Metric": metric_name,
                "Shift Score (PSI)": f"{d['psi']:.4f}",
                "Shift Severity": d["severity"],
                "Risk Level": d["risk_level"]
            })
        return pd.DataFrame(rows)

    def compute_overall_drift(
        self,
        eval_df: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Aggregates feature-level drift into a standardized vector.
        """
        f_drift = self.compute_feature_drift(eval_df, alpha=alpha)
        
        ks_stats = [v["ks_stat"] for v in f_drift.values() if "ks_stat" in v]
        psis = [v["psi"] for v in f_drift.values() if "psi" in v]
        jsds = [v["js_divergence"] for v in f_drift.values() if "js_divergence" in v]
        drifted_count = sum(1 for v in f_drift.values() if v.get("drift_detected", False))

        return {
            "mean_ks_stat": float(np.mean(ks_stats)) if ks_stats else 0.0,
            "max_ks_stat": float(np.max(ks_stats)) if ks_stats else 0.0,
            "mean_psi": float(np.mean(psis)) if psis else 0.0,
            "max_psi": float(np.max(psis)) if psis else 0.0,
            "mean_js_divergence": float(np.mean(jsds)) if jsds else 0.0,
            "pct_features_shifted": float(drifted_count / (len(f_drift) + 1e-9)),
            "num_features_shifted": float(drifted_count),
            "feature_drift_dict": f_drift
        }
