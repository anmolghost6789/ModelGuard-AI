# tests/test_experiment_manager.py
import os
import unittest
from src.experiment_manager import ExperimentManager


class TestExperimentManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = "storage/test_experiments.db"
        cls.mgr = ExperimentManager(db_path=cls.test_db)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_run_logging_and_retrieval(self):
        run_id = self.mgr.start_new_run(
            dataset_name="TestDataset",
            task_type="classification",
            target_column="label",
            primary_model_name="Random Forest",
            sample_count=1000,
            feature_count=10,
            baseline_metric=0.92,
            notes="Unit test run"
        )
        self.assertTrue(run_id.startswith("run_"))

        # Log a batch
        self.mgr.log_batch_evaluation(
            run_id=run_id,
            condition_name="missing_15",
            quality_score=85.0,
            missing_pct=15.0,
            noise_std=0.0,
            outlier_pct=0.0,
            mean_ks_stat=0.12,
            mean_psi=0.08,
            pct_features_shifted=0.20,
            mean_entropy=0.55,
            mean_confidence=0.78,
            realized_metric=0.88,
            relative_degradation_pct=4.35,
            predicted_failure_prob=0.25,
            alert_status="NORMAL",
            ground_truth_failure=0
        )

        df_runs = self.mgr.get_all_runs()
        self.assertGreaterEqual(len(df_runs), 1)

        df_batches = self.mgr.get_run_batches(run_id)
        self.assertEqual(len(df_batches), 1)
        self.assertEqual(df_batches["condition_name"].iloc[0], "missing_15")
        self.assertEqual(df_batches["alert_status"].iloc[0], "NORMAL")


if __name__ == "__main__":
    unittest.main()
