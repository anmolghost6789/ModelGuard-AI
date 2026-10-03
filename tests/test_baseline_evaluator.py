# tests/test_baseline_evaluator.py
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from src.baseline_evaluator import BaselineEvaluator


class TestBaselineEvaluator(unittest.TestCase):
    def test_classification_metrics(self):
        evaluator = BaselineEvaluator(task_type="classification")
        y_true = pd.Series([1, 0, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.65, 0.3, 0.4, 0.75, 0.2])

        metrics = evaluator.evaluate_classification(y_true, y_prob)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"].shape, (2, 2))
        self.assertGreaterEqual(metrics["brier_score"], 0.0)

    def test_regression_metrics(self):
        evaluator = BaselineEvaluator(task_type="regression")
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 31.0, 39.0])

        metrics = evaluator.evaluate_regression(y_true, y_pred)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertGreater(metrics["r2"], 0.9)

    def test_model_zoo_benchmark(self):
        evaluator = BaselineEvaluator(task_type="classification")
        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "f2": [2, 1, 4, 3, 6, 5]})
        y = pd.Series([0, 0, 0, 1, 1, 1])

        clf1 = LogisticRegression().fit(X, y)
        models = {"LogReg": clf1}
        df_bench = evaluator.benchmark_model_zoo(models, X, y)

        self.assertEqual(len(df_bench), 1)
        self.assertIn("F1-Score", df_bench.columns)
        self.assertIn("ROC-AUC", df_bench.columns)


if __name__ == "__main__":
    unittest.main()
