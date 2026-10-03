# src/stress_engine.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy

NUMERICAL_COLS = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL_COLS = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]


class StressTestEngine:
    """
    Applies controlled perturbations to evaluation datasets strictly
    without touching training data or labels.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def apply_missingness(
        self,
        X: pd.DataFrame,
        cols: Optional[List[str]] = None,
        missing_rate: float = 0.15,
        impute_strategy: str = "median",
        train_medians: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Injects MCAR (Missing Completely at Random) and applies standard imputation.
        """
        X_stressed = X.copy()
        target_cols = cols or NUMERICAL_COLS
        n_rows = len(X_stressed)
        
        mask = self.rng.rand(n_rows, len(target_cols)) < missing_rate
        corrupted_cells = int(mask.sum())
        
        for i, col in enumerate(target_cols):
            col_mask = mask[:, i]
            X_stressed.loc[col_mask, col] = np.nan
            
            # Impute using training median or evaluation median
            fill_val = train_medians[col] if (train_medians is not None and col in train_medians) else X[col].median()
            X_stressed[col] = X_stressed[col].fillna(fill_val)
            
        meta = {
            "type": "missingness",
            "rate": missing_rate,
            "corrupted_cells": corrupted_cells,
            "total_cells": n_rows * len(target_cols),
            "effective_pct": float(corrupted_cells / (n_rows * len(target_cols) + 1e-9))
        }
        return X_stressed, meta

    def apply_gaussian_noise(
        self,
        X: pd.DataFrame,
        cols: Optional[List[str]] = None,
        noise_std: float = 0.20
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Injects zero-mean Gaussian sensor noise.
        """
        X_stressed = X.copy()
        target_cols = cols or NUMERICAL_COLS
        
        for col in target_cols:
            noise = self.rng.normal(0.0, noise_std, size=len(X_stressed))
            X_stressed[col] = X_stressed[col] + noise
            # Clip between observed bounds if normalized [0, 1]
            if col in ["temp", "atemp", "hum"]:
                X_stressed[col] = np.clip(X_stressed[col], 0.0, 1.2)
                
        meta = {
            "type": "noise",
            "noise_std": noise_std,
            "features_affected": len(target_cols)
        }
        return X_stressed, meta

    def apply_outliers(
        self,
        X: pd.DataFrame,
        cols: Optional[List[str]] = None,
        outlier_rate: float = 0.10,
        magnitude: float = 3.5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Injects sudden extreme value sensor spikes.
        """
        X_stressed = X.copy()
        target_cols = cols or NUMERICAL_COLS
        n_rows = len(X_stressed)
        
        mask = self.rng.rand(n_rows, len(target_cols)) < outlier_rate
        for i, col in enumerate(target_cols):
            col_mask = mask[:, i]
            col_std = X[col].std() if X[col].std() > 0 else 0.1
            direction = self.rng.choice([-1.0, 1.0], size=int(col_mask.sum()))
            spikes = direction * magnitude * col_std
            X_stressed.loc[col_mask, col] = X_stressed.loc[col_mask, col] + spikes
            
        meta = {
            "type": "outliers",
            "outlier_rate": outlier_rate,
            "magnitude": magnitude,
            "affected_count": int(mask.sum())
        }
        return X_stressed, meta

    def apply_covariate_shift(
        self,
        X: pd.DataFrame,
        shifts: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Systematic feature distribution shift (e.g. sustained heatwave, dry weather).
        """
        X_stressed = X.copy()
        default_shifts = {"temp": 0.20, "atemp": 0.20, "hum": -0.25}
        active_shifts = shifts or default_shifts
        
        for col, delta in active_shifts.items():
            if col in X_stressed.columns:
                X_stressed[col] = X_stressed[col] + delta
                if col in ["temp", "atemp", "hum"]:
                    X_stressed[col] = np.clip(X_stressed[col], 0.0, 1.2)
                    
        meta = {
            "type": "covariate_shift",
            "shifts": active_shifts
        }
        return X_stressed, meta

    def apply_categorical_shift(
        self,
        X: pd.DataFrame,
        col: str = "weathersit",
        target_value: int = 3,
        inflation_rate: float = 0.30
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Distorts categorical frequency (e.g. severe rain/snow conditions).
        """
        X_stressed = X.copy()
        n_rows = len(X_stressed)
        mask = self.rng.rand(n_rows) < inflation_rate
        X_stressed.loc[mask, col] = target_value
        
        meta = {
            "type": "categorical_shift",
            "col": col,
            "target_value": target_value,
            "inflation_rate": inflation_rate
        }
        return X_stressed, meta

    def apply_class_imbalance_shift(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pos_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Resamples evaluation instances to distort positive/negative demand ratio.
        """
        pos_idx = y[y == 1].index
        neg_idx = y[y == 0].index
        
        # Desired ratio
        total_eval = len(y)
        n_pos = int(total_eval * pos_ratio)
        n_neg = total_eval - n_pos
        
        sample_pos = self.rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=True)
        sample_neg = self.rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=True)
        
        combined_idx = np.concatenate([sample_pos, sample_neg])
        self.rng.shuffle(combined_idx)
        
        X_resampled = X.loc[combined_idx].reset_index(drop=True)
        y_resampled = y.loc[combined_idx].reset_index(drop=True)
        
        meta = {
            "type": "class_imbalance",
            "original_pos_ratio": float(y.mean()),
            "resampled_pos_ratio": float(y_resampled.mean())
        }
        return X_resampled, y_resampled, meta

    def apply_compound_stress(
        self,
        X: pd.DataFrame,
        missing_rate: float = 0.15,
        noise_std: float = 0.20,
        temp_shift: float = 0.15,
        weather_inflation: float = 0.20,
        train_medians: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies joint compound data quality degradation + distribution shift.
        """
        X1, m1 = self.apply_covariate_shift(X, shifts={"temp": temp_shift, "hum": -temp_shift})
        X2, m2 = self.apply_gaussian_noise(X1, noise_std=noise_std)
        X3, m3 = self.apply_missingness(X2, missing_rate=missing_rate, train_medians=train_medians)
        X4, m4 = self.apply_categorical_shift(X3, col="weathersit", target_value=3, inflation_rate=weather_inflation)
        
        meta = {
            "type": "compound",
            "components": [m1, m2, m3, m4]
        }
        return X4, meta
