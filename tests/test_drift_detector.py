# tests/test_drift_detector.py
import unittest
import numpy as np
import pandas as pd
from src.drift_detector import calculate_psi, calculate_js_divergence, DistributionShiftDetector


class TestDriftDetector(unittest.TestCase):
    def test_psi_identical_vs_shifted(self):
        np.random.seed(42)
        base = np.random.normal(0, 1, 1000)
        same = np.random.normal(0, 1, 1000)
        shifted = np.random.normal(2.5, 1, 1000)

        psi_same = calculate_psi(base, same)
        psi_shifted = calculate_psi(base, shifted)

        self.assertLess(psi_same, 0.10)
        self.assertGreater(psi_shifted, 0.25)

    def test_jsd_categorical(self):
        s1 = pd.Series(["A"] * 50 + ["B"] * 50)
        s2 = pd.Series(["A"] * 50 + ["B"] * 50)
        s3 = pd.Series(["A"] * 95 + ["B"] * 5)

        jsd_same = calculate_js_divergence(s1, s2)
        jsd_diff = calculate_js_divergence(s1, s3)

        self.assertAlmostEqual(jsd_same, 0.0, places=3)
        self.assertGreater(jsd_diff, 0.10)

    def test_feature_drift_table(self):
        ref_df = pd.DataFrame({
            "temp": np.random.normal(20, 5, 500),
            "weathersit": np.random.choice([1, 2, 3], size=500)
        })
        eval_df = pd.DataFrame({
            "temp": np.random.normal(35, 5, 500),  # Extreme heatwave shift
            "weathersit": np.random.choice([1, 2, 3], size=500)
        })

        detector = DistributionShiftDetector(ref_df, numerical_cols=["temp"], categorical_cols=["weathersit"])
        tbl = detector.get_feature_drift_table(eval_df)

        self.assertEqual(len(tbl), 2)
        self.assertIn("Shift Severity", tbl.columns)
        self.assertIn("Risk Level", tbl.columns)

        # Temp should be high risk
        temp_row = tbl[tbl["Feature"] == "temp"].iloc[0]
        self.assertEqual(temp_row["Risk Level"], "High")


if __name__ == "__main__":
    unittest.main()
