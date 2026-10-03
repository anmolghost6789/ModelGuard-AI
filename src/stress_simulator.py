# src/stress_simulator.py
"""
MODULE 7: Controlled Failure Simulation Laboratory
Applies controlled perturbations (missingness, outliers, noise, imbalance,
covariate drift, categorical drift, compound stress) and measures model degradation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from src.stress_engine import StressTestEngine


class ControlledFailureSimulator:
    """
    Experimental perturbation testbed for machine learning models.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.engine = StressTestEngine(random_state=random_state)

    def run_stress_suite(
        self,
        eval_fn: Any,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        baseline_score: float,
        train_medians: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Executes standard stress perturbations and measures performance degradation.
        eval_fn: callable that takes (X, y) and returns a dict with 'f1' or primary metric.
        """
        results = []

        # 0. Clean Baseline Reference
        clean_res = eval_fn(X_eval, y_eval)
        clean_metric = clean_res.get("f1", clean_res.get("r2", baseline_score))
        results.append({
            "Perturbation Type": "Clean Baseline",
            "Severity / Parameter": "None",
            "Observed Metric": clean_metric,
            "Degradation (Delta)": 0.0,
            "Relative Degradation %": 0.0,
            "Status": "Normal"
        })

        # 1. Missingness Conditions
        for rate in [0.05, 0.15, 0.30]:
            X_s, _ = self.engine.apply_missingness(X_eval, missing_rate=rate, train_medians=train_medians)
            r = eval_fn(X_s, y_eval)
            m = r.get("f1", r.get("r2", 0.0))
            delta = clean_metric - m
            results.append({
                "Perturbation Type": "Missing Values (MCAR)",
                "Severity / Parameter": f"{int(rate*100)}% Missing",
                "Observed Metric": m,
                "Degradation (Delta)": delta,
                "Relative Degradation %": (delta / (clean_metric + 1e-9)) * 100,
                "Status": "Degraded" if delta > 0.05 else "Tolerated"
            })

        # 2. Gaussian Noise
        for std in [0.15, 0.30, 0.50]:
            X_s, _ = self.engine.apply_gaussian_noise(X_eval, noise_std=std)
            r = eval_fn(X_s, y_eval)
            m = r.get("f1", r.get("r2", 0.0))
            delta = clean_metric - m
            results.append({
                "Perturbation Type": "Gaussian Sensor Noise",
                "Severity / Parameter": f"Noise std={std}",
                "Observed Metric": m,
                "Degradation (Delta)": delta,
                "Relative Degradation %": (delta / (clean_metric + 1e-9)) * 100,
                "Status": "Degraded" if delta > 0.05 else "Tolerated"
            })

        # 3. Outlier Spikes
        for rate in [0.05, 0.15]:
            X_s, _ = self.engine.apply_outliers(X_eval, outlier_rate=rate, magnitude=4.0)
            r = eval_fn(X_s, y_eval)
            m = r.get("f1", r.get("r2", 0.0))
            delta = clean_metric - m
            results.append({
                "Perturbation Type": "Outlier Spikes (+4σ)",
                "Severity / Parameter": f"{int(rate*100)}% Contamination",
                "Observed Metric": m,
                "Degradation (Delta)": delta,
                "Relative Degradation %": (delta / (clean_metric + 1e-9)) * 100,
                "Status": "Degraded" if delta > 0.05 else "Tolerated"
            })

        # 4. Covariate Drift
        shifts = {"temp": -0.25, "atemp": -0.25, "hum": 0.15}
        X_s, _ = self.engine.apply_covariate_shift(X_eval, shifts=shifts)
        r = eval_fn(X_s, y_eval)
        m = r.get("f1", r.get("r2", 0.0))
        delta = clean_metric - m
        results.append({
            "Perturbation Type": "Directional Covariate Shift",
            "Severity / Parameter": "Cold Snap (Δtemp=-0.25)",
            "Observed Metric": m,
            "Degradation (Delta)": delta,
            "Relative Degradation %": (delta / (clean_metric + 1e-9)) * 100,
            "Status": "Failure" if delta / (clean_metric + 1e-9) >= 0.15 else "Degraded"
        })

        # 5. Compound Multi-Stress
        X_s, _ = self.engine.apply_compound_stress(
            X_eval, missing_rate=0.15, noise_std=0.25, temp_shift=0.20, weather_inflation=0.25, train_medians=train_medians
        )
        r = eval_fn(X_s, y_eval)
        m = r.get("f1", r.get("r2", 0.0))
        delta = clean_metric - m
        results.append({
            "Perturbation Type": "Compound Stress",
            "Severity / Parameter": "Missing + Noise + Drift",
            "Observed Metric": m,
            "Degradation (Delta)": delta,
            "Relative Degradation %": (delta / (clean_metric + 1e-9)) * 100,
            "Status": "Failure" if delta / (clean_metric + 1e-9) >= 0.15 else "Degraded"
        })

        return pd.DataFrame(results)
