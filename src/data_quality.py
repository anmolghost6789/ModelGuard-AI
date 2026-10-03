# src/data_quality.py
"""
MODULE 2: Data Quality Analyzer Engine
Quantifies dataset health through completeness, uniqueness, plausibility,
feature variance, and class balance, computing an overall Data Quality Score (0-100).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class DataQualityAnalyzer:
    """
    Comprehensive data quality auditor for tabular machine learning pipelines.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        task_type: str = "classification"
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.task_type = task_type
        self.n_rows, self.n_cols = df.shape
        self.feature_cols = [c for c in df.columns if c != target_col]

    def compute_completeness(self) -> Dict[str, Any]:
        """Calculates missingness metrics and completeness score (0-100)."""
        missing_per_col = self.df.isnull().sum()
        total_missing = int(missing_per_col.sum())
        total_cells = self.n_rows * self.n_cols
        missing_rate_overall = total_missing / (total_cells + 1e-9)

        col_missing_pct = (missing_per_col / (self.n_rows + 1e-9) * 100).to_dict()
        severely_missing_cols = [c for c, p in col_missing_pct.items() if p > 20.0]

        # Completeness score: 100 is fully complete
        score = max(0.0, 100.0 * (1.0 - missing_rate_overall * 2.0))
        return {
            "score": float(score),
            "total_missing_cells": total_missing,
            "overall_missing_pct": float(missing_rate_overall * 100),
            "col_missing_pct": col_missing_pct,
            "severely_missing_cols": severely_missing_cols
        }

    def compute_uniqueness(self) -> Dict[str, Any]:
        """Calculates duplicate records and uniqueness score (0-100)."""
        n_duplicates = int(self.df.duplicated().sum())
        duplicate_rate = n_duplicates / (self.n_rows + 1e-9)

        score = max(0.0, 100.0 * (1.0 - duplicate_rate * 3.0))
        return {
            "score": float(score),
            "duplicate_rows": n_duplicates,
            "duplicate_pct": float(duplicate_rate * 100)
        }

    def compute_plausibility_and_outliers(self) -> Dict[str, Any]:
        """Scans continuous features for outliers and extreme bounds."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_feature_cols = [c for c in numeric_cols if c != self.target_col]

        outlier_counts = {}
        total_outlier_cells = 0
        total_num_cells = len(numeric_feature_cols) * self.n_rows

        for col in numeric_feature_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            cnt = int(((s < lower) | (s > upper)).sum())
            outlier_counts[col] = cnt
            total_outlier_cells += cnt

        outlier_rate = total_outlier_cells / (total_num_cells + 1e-9)
        score = max(0.0, 100.0 * (1.0 - outlier_rate * 2.5))

        return {
            "score": float(score),
            "total_outlier_cells": int(total_outlier_cells),
            "overall_outlier_pct": float(outlier_rate * 100),
            "outlier_per_col": outlier_counts
        }

    def compute_variance_health(self) -> Dict[str, Any]:
        """Checks for zero-variance or near-zero variance features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_feature_cols = [c for c in numeric_cols if c != self.target_col]

        zero_var_cols = []
        low_var_cols = []
        variances = {}

        for col in numeric_feature_cols:
            v = float(self.df[col].var())
            variances[col] = v
            if v == 0.0 or np.isnan(v):
                zero_var_cols.append(col)
            elif v < 1e-4:
                low_var_cols.append(col)

        # Penalty for degenerate features
        penalty = (len(zero_var_cols) * 25.0) + (len(low_var_cols) * 10.0)
        score = max(0.0, 100.0 - penalty)

        return {
            "score": float(score),
            "zero_variance_cols": zero_var_cols,
            "low_variance_cols": low_var_cols,
            "variances": variances
        }

    def compute_class_balance(self) -> Dict[str, Any]:
        """Evaluates target class balance or regression skewness."""
        if self.target_col is None or self.target_col not in self.df.columns:
            return {"score": 100.0, "status": "no_target_specified"}

        target_series = self.df[self.target_col].dropna()

        if self.task_type == "classification":
            counts = target_series.value_counts(normalize=True).values
            num_classes = len(counts)

            if num_classes <= 1:
                return {"score": 0.0, "status": "single_class_collapse"}

            # Normalized entropy: 1.0 when perfectly balanced
            norm_entropy = -np.sum(counts * np.log(counts + 1e-12)) / np.log(num_classes)
            score = float(np.clip(norm_entropy * 100.0, 0.0, 100.0))
            return {
                "score": score,
                "class_ratios": self.df[self.target_col].value_counts(normalize=True).to_dict(),
                "normalized_entropy": float(norm_entropy)
            }
        else:
            # Regression skewness check
            skewness = float(abs(target_series.skew()))
            # Ideal skewness is close to 0; heavily skewed penalizes
            score = max(0.0, 100.0 - min(100.0, skewness * 20.0))
            return {
                "score": float(score),
                "skewness": skewness
            }

    def generate_quality_report(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes composite Data Quality Score (0-100) and actionable warnings.
        """
        w = weights or {
            "completeness": 0.30,
            "plausibility": 0.25,
            "uniqueness": 0.15,
            "variance": 0.15,
            "balance": 0.15
        }

        comp = self.compute_completeness()
        uniq = self.compute_uniqueness()
        plaus = self.compute_plausibility_and_outliers()
        var_h = self.compute_variance_health()
        bal = self.compute_class_balance()

        overall_dqs = (
            w["completeness"] * comp["score"] +
            w["plausibility"] * plaus["score"] +
            w["uniqueness"] * uniq["score"] +
            w["variance"] * var_h["score"] +
            w["balance"] * bal["score"]
        )
        overall_dqs = float(np.clip(overall_dqs, 0.0, 100.0))

        # Actionable recommendations
        warnings = []
        if comp["overall_missing_pct"] > 5.0:
            warnings.append(f"Missingness detected: {comp['overall_missing_pct']:.1f}% cells missing. Imputation required.")
        if uniq["duplicate_pct"] > 0.0:
            warnings.append(f"Duplicate rows detected: {uniq['duplicate_rows']} duplicates ({uniq['duplicate_pct']:.2f}%). De-duplication recommended.")
        if plaus["overall_outlier_pct"] > 5.0:
            warnings.append(f"Elevated outlier density: {plaus['overall_outlier_pct']:.1f}% numerical values outside 1.5x IQR.")
        if var_h["zero_variance_cols"]:
            warnings.append(f"Zero-variance features found: {var_h['zero_variance_cols']}. Drop these non-informative columns.")
        if bal["score"] < 60.0:
            warnings.append("Severe target imbalance detected. Resampling or class-weighted loss advised.")

        return {
            "data_quality_score": overall_dqs,
            "grade": "EXCELLENT" if overall_dqs >= 85 else ("ACCEPTABLE" if overall_dqs >= 70 else "POOR"),
            "component_scores": {
                "completeness": comp["score"],
                "plausibility": plaus["score"],
                "uniqueness": uniq["score"],
                "variance": var_h["score"],
                "class_balance": bal["score"]
            },
            "completeness_metrics": comp,
            "uniqueness_metrics": uniq,
            "plausibility_metrics": plaus,
            "variance_metrics": var_h,
            "balance_metrics": bal,
            "warnings": warnings
        }
