# src/experiment_manager.py
"""
MODULE 11: Experiment Management & SQLite Audit Logger
Persists experimental runs, baseline metrics, controlled disturbances,
telemetry snapshots, and failure-risk predictions in an ACID SQLite database.
"""

import os
import sqlite3
import uuid
import pandas as pd
from typing import Dict, Any, List, Optional


class ExperimentManager:
    """
    Manages persistent experiment tracking and telemetry logging in SQLite.
    """
    def __init__(self, db_path: str = "storage/experiments.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize_schema(self):
        """Creates required relational tables if not present."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                dataset_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                target_column TEXT NOT NULL,
                primary_model_name TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                feature_count INTEGER NOT NULL,
                baseline_primary_metric REAL,
                notes TEXT
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_batches (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                condition_name TEXT NOT NULL,
                missing_pct REAL NOT NULL,
                noise_std REAL NOT NULL,
                outlier_pct REAL NOT NULL,
                quality_score REAL NOT NULL,
                mean_ks_stat REAL NOT NULL,
                mean_psi REAL NOT NULL,
                pct_features_shifted REAL NOT NULL,
                mean_entropy REAL NOT NULL,
                mean_confidence REAL NOT NULL,
                realized_metric REAL,
                relative_degradation_pct REAL,
                predicted_failure_prob REAL NOT NULL,
                predicted_risk_score REAL NOT NULL,
                alert_status TEXT NOT NULL,
                ground_truth_failure INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id)
            );
            """)
            conn.commit()
        finally:
            conn.close()

    def start_new_run(
        self,
        dataset_name: str,
        task_type: str,
        target_column: str,
        primary_model_name: str,
        sample_count: int,
        feature_count: int,
        baseline_metric: float,
        notes: str = ""
    ) -> str:
        """Logs a new experimental session and returns unique run_id."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO experiment_runs (
                run_id, dataset_name, task_type, target_column, primary_model_name,
                sample_count, feature_count, baseline_primary_metric, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, dataset_name, task_type, target_column, primary_model_name,
                sample_count, feature_count, float(baseline_metric), notes
            ))
            conn.commit()
        finally:
            conn.close()
        return run_id

    def log_batch_evaluation(
        self,
        run_id: str,
        condition_name: str,
        quality_score: float,
        missing_pct: float,
        noise_std: float,
        outlier_pct: float,
        mean_ks_stat: float,
        mean_psi: float,
        pct_features_shifted: float,
        mean_entropy: float,
        mean_confidence: float,
        realized_metric: float,
        relative_degradation_pct: float,
        predicted_failure_prob: float,
        alert_status: str,
        ground_truth_failure: int
    ):
        """Logs individual evaluation condition results."""
        risk_score = float(predicted_failure_prob * 100.0)
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO evaluation_batches (
                run_id, condition_name, missing_pct, noise_std, outlier_pct, quality_score,
                mean_ks_stat, mean_psi, pct_features_shifted, mean_entropy, mean_confidence,
                realized_metric, relative_degradation_pct, predicted_failure_prob,
                predicted_risk_score, alert_status, ground_truth_failure
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, condition_name, float(missing_pct), float(noise_std), float(outlier_pct), float(quality_score),
                float(mean_ks_stat), float(mean_psi), float(pct_features_shifted), float(mean_entropy), float(mean_confidence),
                float(realized_metric), float(relative_degradation_pct), float(predicted_failure_prob),
                float(risk_score), alert_status, int(ground_truth_failure)
            ))
            conn.commit()
        finally:
            conn.close()

    def get_all_runs(self) -> pd.DataFrame:
        """Retrieves history of all experimental runs."""
        conn = self._get_connection()
        try:
            df = pd.read_sql_query("SELECT * FROM experiment_runs ORDER BY timestamp DESC", conn)
        finally:
            conn.close()
        return df

    def get_run_batches(self, run_id: str) -> pd.DataFrame:
        """Retrieves all logged batches for a given experiment run."""
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM evaluation_batches WHERE run_id = ? ORDER BY batch_id ASC",
                conn,
                params=(run_id,)
            )
        finally:
            conn.close()
        return df
