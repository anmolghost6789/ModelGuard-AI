# tests/test_data_profiler.py
import unittest
import pandas as pd
from src.data_profiler import DatasetProfiler


class TestDatasetProfiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiler = DatasetProfiler.from_csv("hour.csv")
        cls.profile = cls.profiler.generate_full_profile()

    def test_dimensions(self):
        self.assertEqual(self.profile["shape"]["rows"], 17379)
        self.assertEqual(self.profile["shape"]["cols"], 17)

    def test_column_types(self):
        types = self.profile["column_types"]
        self.assertEqual(types["instant"], "identifier")
        self.assertEqual(types["dteday"], "datetime")
        self.assertIn("temp", self.profile["numerical_columns"])
        self.assertIn("season", self.profile["categorical_columns"])

    def test_target_recommendation(self):
        self.assertEqual(self.profile["recommended_target"], "cnt")
        self.assertEqual(self.profile["inferred_problem_type"], "regression")

    def test_missing_and_duplicates(self):
        self.assertEqual(self.profile["total_missing_cells"], 0)
        self.assertEqual(self.profile["duplicate_rows"], 0)

    def test_outlier_detection(self):
        outliers = self.profile["outlier_summary"]
        self.assertIn("temp", outliers)
        self.assertIn("hum", outliers)
        self.assertGreaterEqual(outliers["hum"]["outlier_count"], 0)


if __name__ == "__main__":
    unittest.main()
