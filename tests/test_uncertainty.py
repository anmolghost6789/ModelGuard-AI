# tests/test_uncertainty.py
import unittest
import numpy as np
import pandas as pd
from src.uncertainty import (
    compute_instance_uncertainty,
    compute_expected_calibration_error,
    compute_batch_uncertainty_summary,
    audit_prediction_samples
)


class TestUncertaintyEngine(unittest.TestCase):
    def test_instance_uncertainty_bounds(self):
        # 3 samples: confident 1, confident 0, completely ambiguous (50/50)
        probs = np.array([
            [0.05, 0.95],
            [0.98, 0.02],
            [0.50, 0.50]
        ])
        df_unc = compute_instance_uncertainty(probs)

        self.assertEqual(len(df_unc), 3)
        self.assertAlmostEqual(df_unc["confidence"].iloc[0], 0.95, places=2)
        self.assertAlmostEqual(df_unc["confidence"].iloc[2], 0.50, places=2)

        # Ambiguous sample must have maximum entropy (~1.0)
        self.assertAlmostEqual(df_unc["entropy"].iloc[2], 1.0, places=2)
        # Confident sample must have very low entropy (~0.0)
        self.assertLess(df_unc["entropy"].iloc[1], 0.20)

    def test_calibration_error(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.85, 0.95, 0.1, 0.05, 0.15])
        ece, df_bins = compute_expected_calibration_error(y_true, y_prob, n_bins=5)

        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertEqual(len(df_bins), 5)

    def test_audit_prediction_samples(self):
        probs = np.array([
            [0.1, 0.9],   # High confidence
            [0.48, 0.52]  # Ambiguous (high uncertainty)
        ])
        audit_df = audit_prediction_samples(probs, class_labels=["Normal", "Fraud"])

        self.assertEqual(len(audit_df), 2)
        self.assertEqual(audit_df["Uncertainty Level"].iloc[0], "Low")
        self.assertEqual(audit_df["Action Recommendation"].iloc[0], "Approved")

        self.assertEqual(audit_df["Uncertainty Level"].iloc[1], "High")
        self.assertEqual(audit_df["Action Recommendation"].iloc[1], "Prediction should be reviewed")


if __name__ == "__main__":
    unittest.main()
