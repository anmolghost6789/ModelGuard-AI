# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional

FEATURE_COLS = [
    "season",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

NUMERICAL_COLS = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL_COLS = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
LEAKAGE_COLS = ["instant", "casual", "registered"]


def load_raw_dataset(file_path: str = "hour.csv") -> pd.DataFrame:
    """Loads raw bike sharing dataset and performs basic validation."""
    df = pd.read_csv(file_path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, float]:
    """
    Extracts clean feature matrix X, binary classification target y,
    and regression target cnt.
    Threshold is computed from training median if not provided.
    """
    # Remove leakage columns if present
    drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    clean_df = df.drop(columns=drop_cols)

    cnt = clean_df["cnt"]
    if threshold is None:
        threshold = float(cnt.median())

    y = (cnt >= threshold).astype(int)
    X = clean_df[FEATURE_COLS].copy()
    return X, y, cnt, threshold


def create_chronological_splits(
    df: pd.DataFrame,
    target_threshold: Optional[float] = None
) -> Dict[str, Dict]:
    """
    Creates strict chronological non-leaking partitions:
    - Train: 2011-01-01 to 2011-12-31 (yr == 0)
    - Val:   2012-01-01 to 2012-03-31 (Q1 2012)
    - Test:  2012-04-01 to 2012-06-30 (Q2 2012)
    - Future:2012-07-01 to 2012-12-31 (H2 2012)
    
    Fits scaler ONLY on training data.
    """
    # Filter subsets by date
    train_mask = (df["dteday"] >= "2011-01-01") & (df["dteday"] <= "2011-12-31")
    val_mask   = (df["dteday"] >= "2012-01-01") & (df["dteday"] <= "2012-03-31")
    test_mask  = (df["dteday"] >= "2012-04-01") & (df["dteday"] <= "2012-06-30")
    fut_mask   = (df["dteday"] >= "2012-07-01") & (df["dteday"] <= "2012-12-31")

    train_raw = df[train_mask].copy().reset_index(drop=True)
    val_raw   = df[val_mask].copy().reset_index(drop=True)
    test_raw  = df[test_mask].copy().reset_index(drop=True)
    fut_raw   = df[fut_mask].copy().reset_index(drop=True)

    # Determine threshold from training set median to avoid test leakage
    if target_threshold is None:
        target_threshold = float(train_raw["cnt"].median())

    # Extract features and targets
    X_train, y_train, cnt_train, _ = prepare_features_and_target(train_raw, threshold=target_threshold)
    X_val, y_val, cnt_val, _       = prepare_features_and_target(val_raw, threshold=target_threshold)
    X_test, y_test, cnt_test, _    = prepare_features_and_target(test_raw, threshold=target_threshold)
    X_fut, y_fut, cnt_fut, _       = prepare_features_and_target(fut_raw, threshold=target_threshold)

    # Fit scaler strictly on training set
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=FEATURE_COLS)
    X_val_scaled   = pd.DataFrame(scaler.transform(X_val), columns=FEATURE_COLS)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS)
    X_fut_scaled   = pd.DataFrame(scaler.transform(X_fut), columns=FEATURE_COLS)

    splits = {
        "train": {
            "X": X_train,
            "X_scaled": X_train_scaled,
            "y": y_train,
            "cnt": cnt_train,
            "raw": train_raw,
            "dates": ("2011-01-01", "2011-12-31")
        },
        "val": {
            "X": X_val,
            "X_scaled": X_val_scaled,
            "y": y_val,
            "cnt": cnt_val,
            "raw": val_raw,
            "dates": ("2012-01-01", "2012-03-31")
        },
        "test": {
            "X": X_test,
            "X_scaled": X_test_scaled,
            "y": y_test,
            "cnt": cnt_test,
            "raw": test_raw,
            "dates": ("2012-04-01", "2012-06-30")
        },
        "future": {
            "X": X_fut,
            "X_scaled": X_fut_scaled,
            "y": y_fut,
            "cnt": cnt_fut,
            "raw": fut_raw,
            "dates": ("2012-07-01", "2012-12-31")
        },
        "meta": {
            "threshold": target_threshold,
            "scaler": scaler,
            "feature_cols": FEATURE_COLS,
            "numerical_cols": NUMERICAL_COLS,
            "categorical_cols": CATEGORICAL_COLS
        }
    }
    return splits
