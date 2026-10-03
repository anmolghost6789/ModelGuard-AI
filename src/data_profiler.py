# src/data_profiler.py
"""
MODULE 1: Dataset Ingestion & Automated Schema Profiler
Handles arbitrary CSV ingestion, type inference, target recommendation,
duplicate detection, outlier scanning, and statistical profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class DatasetProfiler:
    """
    Automated profiler for tabular machine learning datasets.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.n_rows, self.n_cols = df.shape

    @classmethod
    def from_csv(cls, file_path_or_buffer: Any) -> "DatasetProfiler":
        """Factory method to load CSV from file path or uploaded buffer."""
        df = pd.read_csv(file_path_or_buffer)
        return cls(df)

    def detect_column_types(self) -> Dict[str, str]:
        """
        Infers semantic types: 'datetime', 'identifier', 'categorical', 'numerical'.
        """
        col_types = {}
        for col in self.df.columns:
            s = self.df[col]
            n_unique = s.nunique(dropna=True)

            # Check if identifier (unique on every row or matches id pattern)
            if (n_unique == self.n_rows and self.n_rows > 50) or col.lower() in ["id", "instant", "uuid", "index"]:
                col_types[col] = "identifier"
                continue

            # Check if date/time string
            if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s) or "date" in col.lower() or "time" in col.lower():
                try:
                    pd.to_datetime(s.dropna().iloc[:100], errors="raise")
                    col_types[col] = "datetime"
                    continue
                except Exception:
                    pass

            # Numerical vs Categorical
            if pd.api.types.is_numeric_dtype(s):
                if n_unique <= 10 and not pd.api.types.is_float_dtype(s):
                    col_types[col] = "categorical"
                else:
                    col_types[col] = "numerical"
            else:
                col_types[col] = "categorical"

        return col_types

    def recommend_target_column(self) -> Tuple[Optional[str], str]:
        """
        Recommends primary target column and inferred problem type.
        """
        candidates = ["cnt", "target", "label", "class", "y", "output", "price", "churn", "status", "demand"]
        col_lower = {c.lower(): c for c in self.df.columns}

        target_col = None
        for cand in candidates:
            if cand in col_lower:
                target_col = col_lower[cand]
                break

        if target_col is None:
            # Default to the last column
            target_col = self.df.columns[-1]

        target_series = self.df[target_col].dropna()
        n_unique = target_series.nunique()

        if n_unique == 2:
            problem_type = "binary_classification"
        elif 3 <= n_unique <= 15 and (pd.api.types.is_integer_dtype(target_series) or target_series.dtype == "object"):
            problem_type = "multiclass_classification"
        elif pd.api.types.is_numeric_dtype(target_series):
            problem_type = "regression"
        else:
            problem_type = "binary_classification"

        return target_col, problem_type

    def detect_outliers_iqr(self, numerical_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Calculates outlier counts using the 1.5 * IQR method.
        """
        outlier_summary = {}
        for col in numerical_cols:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = s[(s < lower_bound) | (s > upper_bound)]
            outlier_summary[col] = {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": int(len(outliers)),
                "outlier_pct": float(len(outliers) / (len(s) + 1e-9)) * 100
            }
        return outlier_summary

    def generate_full_profile(self) -> Dict[str, Any]:
        """
        Generates complete comprehensive dataset health summary dictionary.
        """
        col_types = self.detect_column_types()
        num_cols = [c for c, t in col_types.items() if t == "numerical"]
        cat_cols = [c for c, t in col_types.items() if t == "categorical"]
        date_cols = [c for c, t in col_types.items() if t == "datetime"]
        id_cols = [c for c, t in col_types.items() if t == "identifier"]

        rec_target, problem_type = self.recommend_target_column()
        outliers = self.detect_outliers_iqr(num_cols)
        total_outliers = sum(v["outlier_count"] for v in outliers.values())

        # Missing values
        missing_counts = self.df.isnull().sum().to_dict()
        total_missing_cells = sum(missing_counts.values())
        missing_pct_overall = (total_missing_cells / (self.n_rows * self.n_cols + 1e-9)) * 100

        # Duplicate rows
        n_duplicates = int(self.df.duplicated().sum())

        # Class distribution if classification
        class_dist = {}
        if rec_target and problem_type != "regression":
            class_dist = self.df[rec_target].value_counts(normalize=True).to_dict()

        # Numeric stats summary
        numeric_stats = {}
        for c in num_cols:
            s = self.df[c].dropna()
            numeric_stats[c] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median())
            }

        return {
            "shape": {"rows": self.n_rows, "cols": self.n_cols},
            "column_types": col_types,
            "categorical_columns": cat_cols,
            "numerical_columns": num_cols,
            "datetime_columns": date_cols,
            "identifier_columns": id_cols,
            "recommended_target": rec_target,
            "inferred_problem_type": problem_type,
            "total_missing_cells": int(total_missing_cells),
            "missing_pct_overall": float(missing_pct_overall),
            "missing_per_column": missing_counts,
            "duplicate_rows": n_duplicates,
            "outlier_summary": outliers,
            "total_outliers": int(total_outliers),
            "class_distribution": class_dist,
            "numeric_stats": numeric_stats
        }
