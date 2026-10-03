# src/failure_dataset.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from src.stress_engine import StressTestEngine
from src.drift_detector import DistributionShiftDetector
from src.uncertainty import compute_batch_uncertainty_summary
from src.baseline_models import evaluate_classifier


def generate_failure_meta_dataset(
    primary_model: Any,
    X_train_ref: pd.DataFrame,
    X_eval_pool: pd.DataFrame,
    y_eval_pool: pd.Series,
    baseline_metrics: Dict[str, float],
    scaler: Any,
    num_samples: int = 250,
    window_size: int = 200,
    failure_threshold_rel: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a secondary experimental meta-dataset where each row represents
    an evaluation batch/condition, and the target is whether the ML model
    experienced a performance failure (e.g. F1 drop >= failure_threshold_rel).
    """
    rng = np.random.RandomState(random_state)
    stress_engine = StressTestEngine(random_state=random_state)
    drift_detector = DistributionShiftDetector(reference_df=X_train_ref)

    train_medians = X_train_ref.median()
    baseline_f1 = baseline_metrics.get("f1", baseline_metrics.get("F1-Score", 0.94))
    baseline_acc = baseline_metrics.get("accuracy", baseline_metrics.get("Accuracy", 0.92))

    records = []
    total_eval_len = len(X_eval_pool)

    # Condition distribution: 20% clean, 15% missing, 15% noise, 10% outlier, 15% drift, 10% imbalance, 15% compound
    conditions = (
        ["clean"] * int(num_samples * 0.20) +
        ["missing"] * int(num_samples * 0.15) +
        ["noise"] * int(num_samples * 0.15) +
        ["outlier"] * int(num_samples * 0.10) +
        ["drift"] * int(num_samples * 0.15) +
        ["imbalance"] * int(num_samples * 0.10) +
        ["compound"] * (num_samples - int(num_samples * 0.85))
    )
    rng.shuffle(conditions)

    for i, cond in enumerate(conditions):
        # Sample a contiguous or random window from evaluation pool
        if total_eval_len > window_size:
            start_idx = rng.randint(0, total_eval_len - window_size)
            idx_range = list(range(start_idx, start_idx + window_size))
        else:
            idx_range = list(range(total_eval_len))

        X_batch = X_eval_pool.iloc[idx_range].copy().reset_index(drop=True)
        y_batch = y_eval_pool.iloc[idx_range].copy().reset_index(drop=True)

        # Perturbation variables
        missing_rate = 0.0
        noise_std = 0.0
        outlier_rate = 0.0
        cov_shift_mag = 0.0
        corrupt_features = 0

        # Apply condition
        if cond == "clean":
            X_mod = X_batch.copy()
            y_mod = y_batch.copy()

        elif cond == "missing":
            missing_rate = rng.uniform(0.05, 0.35)
            corrupt_features = rng.randint(1, 4)
            X_mod, _ = stress_engine.apply_missingness(
                X_batch,
                missing_rate=missing_rate,
                train_medians=train_medians
            )
            y_mod = y_batch

        elif cond == "noise":
            noise_std = rng.uniform(0.08, 0.45)
            corrupt_features = 4
            X_mod, _ = stress_engine.apply_gaussian_noise(
                X_batch,
                noise_std=noise_std
            )
            y_mod = y_batch

        elif cond == "outlier":
            outlier_rate = rng.uniform(0.04, 0.22)
            corrupt_features = rng.randint(1, 4)
            X_mod, _ = stress_engine.apply_outliers(
                X_batch,
                outlier_rate=outlier_rate,
                magnitude=rng.uniform(3.0, 5.0)
            )
            y_mod = y_batch

        elif cond == "drift":
            shift_temp = rng.uniform(-0.25, 0.35)
            shift_hum = rng.uniform(-0.30, 0.25)
            cov_shift_mag = abs(shift_temp) + abs(shift_hum)
            corrupt_features = 2
            X_mod, _ = stress_engine.apply_covariate_shift(
                X_batch,
                shifts={"temp": shift_temp, "hum": shift_hum}
            )
            y_mod = y_batch

        elif cond == "imbalance":
            pos_ratio = rng.uniform(0.10, 0.85)
            X_mod, y_mod, _ = stress_engine.apply_class_imbalance_shift(
                X_batch, y_batch, pos_ratio=pos_ratio
            )

        elif cond == "compound":
            missing_rate = rng.uniform(0.05, 0.25)
            noise_std = rng.uniform(0.08, 0.30)
            shift_val = rng.uniform(0.08, 0.25)
            cov_shift_mag = shift_val
            corrupt_features = 4
            X_mod, _ = stress_engine.apply_compound_stress(
                X_batch,
                missing_rate=missing_rate,
                noise_std=noise_std,
                temp_shift=shift_val,
                weather_inflation=rng.uniform(0.1, 0.3),
                train_medians=train_medians
            )
            y_mod = y_batch

        # 1. Compute Data Quality Score in [0, 1] (1 = perfect, 0 = completely corrupted)
        dq_penalty = (missing_rate * 1.5) + (noise_std * 1.2) + (outlier_rate * 2.0)
        data_quality_score = max(0.0, 1.0 - min(1.0, dq_penalty))

        # 2. Compute Distribution Shift Metrics (compared to training reference)
        drift_metrics = drift_detector.compute_overall_drift(X_mod)

        # 3. Model Inference & Uncertainty
        X_scaled = pd.DataFrame(scaler.transform(X_mod), columns=X_mod.columns)
        eval_res = evaluate_classifier(primary_model, X_scaled, y_mod)
        unc_metrics = compute_batch_uncertainty_summary(eval_res["y_prob"], model=primary_model, X=X_scaled)

        # 4. Actual Observed Batch Performance
        batch_f1 = eval_res["f1"]
        batch_acc = eval_res["accuracy"]
        f1_degradation = baseline_f1 - batch_f1
        f1_degradation_rel = (baseline_f1 - batch_f1) / (baseline_f1 + 1e-9)
        acc_degradation = baseline_acc - batch_acc

        # 5. Research Ground Truth Failure Label
        # Primary Definition: relative F1 degradation >= failure_threshold_rel
        is_failure = int(f1_degradation_rel >= failure_threshold_rel)

        # Record Meta-Sample
        sample_meta = {
            # Metadata
            "sample_id": i,
            "condition": cond,
            "batch_size": len(X_mod),

            # Group 1: Data Quality Indicators
            "missing_pct": float(missing_rate),
            "noise_std": float(noise_std),
            "outlier_pct": float(outlier_rate),
            "cov_shift_mag": float(cov_shift_mag),
            "corrupt_feature_cnt": float(corrupt_features),
            "data_quality_score": float(data_quality_score),

            # Group 2: Distribution Shift Indicators
            "mean_ks_stat": float(drift_metrics["mean_ks_stat"]),
            "max_ks_stat": float(drift_metrics["max_ks_stat"]),
            "mean_psi": float(drift_metrics["mean_psi"]),
            "max_psi": float(drift_metrics["max_psi"]),
            "mean_js_divergence": float(drift_metrics["mean_js_divergence"]),
            "pct_features_shifted": float(drift_metrics["pct_features_shifted"]),
            "num_features_shifted": float(drift_metrics["num_features_shifted"]),

            # Group 3: Uncertainty & Confidence Indicators
            "mean_confidence": float(unc_metrics["mean_confidence"]),
            "min_confidence": float(unc_metrics["min_confidence"]),
            "mean_entropy": float(unc_metrics["mean_entropy"]),
            "max_entropy": float(unc_metrics["max_entropy"]),
            "pct_high_entropy": float(unc_metrics["pct_high_entropy"]),
            "pct_low_confidence": float(unc_metrics["pct_low_confidence"]),
            "mean_margin": float(unc_metrics["mean_margin"]),
            "mean_ensemble_variance": float(unc_metrics["mean_ensemble_variance"]),

            # Baseline References
            "baseline_f1": float(baseline_f1),
            "baseline_accuracy": float(baseline_acc),

            # Primary Performance Realized (for evaluation / analysis)
            "batch_f1": float(batch_f1),
            "batch_accuracy": float(batch_acc),
            "f1_degradation": float(f1_degradation),
            "f1_degradation_rel": float(f1_degradation_rel),
            "acc_degradation": float(acc_degradation),

            # Alternative Threshold Targets (for sensitivity analysis)
            "failure_thresh_10": int(f1_degradation_rel >= 0.10),
            "failure_thresh_15": int(f1_degradation_rel >= 0.15),
            "failure_thresh_20": int(f1_degradation_rel >= 0.20),
            "failure_thresh_25": int(f1_degradation_rel >= 0.25),

            # Primary Target
            "model_failure": is_failure
        }
        records.append(sample_meta)

    df_meta = pd.DataFrame(records)
    summary = {
        "total_meta_samples": len(df_meta),
        "failure_count": int(df_meta["model_failure"].sum()),
        "failure_rate": float(df_meta["model_failure"].mean()),
        "failure_threshold_rel": failure_threshold_rel,
        "conditions_breakdown": df_meta["condition"].value_counts().to_dict()
    }
    return df_meta, summary
