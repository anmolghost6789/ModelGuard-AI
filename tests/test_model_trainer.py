# tests/test_model_trainer.py
import os
import unittest
import numpy as np
import pandas as pd
from src.model_trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small synthetic dataset for fast test execution
        np.random.seed(42)
        n = 300
        cls.X_train = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.uniform(0, 10, n),
            "f3": np.random.randint(0, 4, n)
        })
        cls.y_class = (cls.X_train["f1"] + cls.X_train["f2"] * 0.5 > 3.0).astype(int)
        cls.y_reg = cls.X_train["f1"] * 2.0 + cls.X_train["f2"] * 1.5 + np.random.randn(n) * 0.2

    def test_classification_training(self):
        trainer = ModelTrainer(task_type="classification", random_state=42)
        models = trainer.get_supported_model_names()
        self.assertIn("Logistic Regression", models)
        self.assertIn("Random Forest", models)

        rf = trainer.train_single_model("Random Forest", self.X_train, self.y_class)
        self.assertTrue(hasattr(rf, "predict"))
        self.assertTrue(hasattr(rf, "predict_proba"))

    def test_regression_training(self):
        trainer = ModelTrainer(task_type="regression", random_state=42)
        models = trainer.get_supported_model_names()
        self.assertIn("Linear Regression", models)
        self.assertIn("Random Forest Regressor", models)

        reg = trainer.train_single_model("Random Forest Regressor", self.X_train, self.y_reg)
        self.assertTrue(hasattr(reg, "predict"))

    def test_save_artifact(self):
        trainer = ModelTrainer(task_type="classification", random_state=42)
        trainer.train_single_model("Decision Tree", self.X_train, self.y_class)
        save_path = "output/models/test_dt.joblib"
        trainer.save_artifact("Decision Tree", save_path)
        self.assertTrue(os.path.exists(save_path))
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    unittest.main()
