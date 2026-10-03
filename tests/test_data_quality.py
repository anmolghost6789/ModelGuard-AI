# tests/test_data_quality.py
import unittest
import numpy as np
import pandas as pd
from src.data_quality import DataQualityAnalyzer


class TestDataQualityAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_raw = pd.read_csv("hour.csv")
        cls.df_raw["high_demand"] = (cls.df_raw["cnt"] >= 109).astype(int)
        cls.analyzer = DataQualityAnalyzer(cls.df_raw, target_col="high_demand", task_type="classification")
        cls.report = cls.analyzer.generate_quality_report()

    def test_clean_data_completeness(self):
        comp = self.report["completeness_metrics"]
        self.assertEqual(comp["total_missing_cells"], 0)
        self.assertEqual(comp["score"], 100.0)

    def test_clean_data_uniqueness(self):
        uniq = self.report["uniqueness_metrics"]
        self.assertEqual(uniq["duplicate_rows"], 0)
        self.assertEqual(uniq["score"], 100.0)

    def test_composite_score_range(self):
        dqs = self.report["data_quality_score"]
        self.assertGreaterEqual(dqs, 0.0)
        self.assertLessEqual(dqs, 100.0)
        self.assertIn(self.report["grade"], ["EXCELLENT", "ACCEPTABLE", "POOR"])

    def test_synthetic_corruption_sensitivity(self):
        df_corrupted = self.df_raw.copy().iloc[:500]
        # Inject 20% missingness into temp and hum
        df_corrupted.loc[:100, "temp"] = np.nan
        df_corrupted.loc[:100, "hum"] = np.nan
        # Inject duplicates
        df_corrupted = pd.concat([df_corrupted, df_corrupted.iloc[:50]], axis=0)

        bad_analyzer = DataQualityAnalyzer(df_corrupted, target_col="high_demand", task_type="classification")
        bad_report = bad_analyzer.generate_quality_report()

        self.assertLess(bad_report["completeness_metrics"]["score"], 100.0)
        self.assertLess(bad_report["uniqueness_metrics"]["score"], 100.0)
        self.assertLess(bad_report["data_quality_score"], self.report["data_quality_score"])
        self.assertGreater(len(bad_report["warnings"]), 0)


if __name__ == "__main__":
    unittest.main()
