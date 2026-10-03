# src/model_trainer.py
"""
MODULE 3: Machine Learning Model Training Engine
Supports multi-architecture model training for classification (LR, DT, RF, XGBoost, SVM)
and regression (Linear, DT, RF, Gradient Boosting) with automatic task routing.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class ModelTrainer:
    """
    Automated trainer and registry for primary classification and regression models.
    """
    def __init__(
        self,
        task_type: str = "classification",
        random_state: int = 42
    ):
        self.task_type = task_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.trained_models: Dict[str, Any] = {}

    def get_supported_model_names(self) -> List[str]:
        """Returns list of supported algorithms for current task type."""
        if self.task_type == "classification":
            models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVM"]
            if HAS_XGB:
                models.append("XGBoost")
            return models
        else:
            models = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"]
            if HAS_XGB:
                models.append("XGBoost Regressor")
            return models

    def instantiate_model(
        self,
        model_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Instantiates an un-fitted model instance with default or custom parameters."""
        p = params or {}
        rs = self.random_state

        if self.task_type == "classification":
            if model_name == "Logistic Regression":
                return LogisticRegression(C=p.get("C", 1.0), max_iter=p.get("max_iter", 1000), random_state=rs)
            elif model_name == "Decision Tree":
                return DecisionTreeClassifier(max_depth=p.get("max_depth", 8), min_samples_leaf=p.get("min_samples_leaf", 4), random_state=rs)
            elif model_name == "Random Forest":
                return RandomForestClassifier(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 12), min_samples_leaf=p.get("min_samples_leaf", 4), random_state=rs, n_jobs=-1)
            elif model_name == "Gradient Boosting":
                return GradientBoostingClassifier(n_estimators=p.get("n_estimators", 100), learning_rate=p.get("learning_rate", 0.1), max_depth=p.get("max_depth", 5), random_state=rs)
            elif model_name == "SVM":
                return SVC(C=p.get("C", 1.0), kernel=p.get("kernel", "rbf"), probability=True, random_state=rs)
            elif model_name == "XGBoost" and HAS_XGB:
                return XGBClassifier(n_estimators=p.get("n_estimators", 100), learning_rate=p.get("learning_rate", 0.1), max_depth=p.get("max_depth", 5), random_state=rs, eval_metric="logloss", n_jobs=-1)
            else:
                raise ValueError(f"Unsupported classification model: {model_name}")

        else:
            # Regression Models
            if model_name == "Linear Regression":
                return Ridge(alpha=p.get("alpha", 1.0), random_state=rs)
            elif model_name == "Decision Tree Regressor":
                return DecisionTreeRegressor(max_depth=p.get("max_depth", 8), min_samples_leaf=p.get("min_samples_leaf", 4), random_state=rs)
            elif model_name == "Random Forest Regressor":
                return RandomForestRegressor(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 12), min_samples_leaf=p.get("min_samples_leaf", 4), random_state=rs, n_jobs=-1)
            elif model_name == "Gradient Boosting Regressor":
                return GradientBoostingRegressor(n_estimators=p.get("n_estimators", 100), learning_rate=p.get("learning_rate", 0.1), max_depth=p.get("max_depth", 5), random_state=rs)
            elif model_name == "XGBoost Regressor" and HAS_XGB:
                return XGBRegressor(n_estimators=p.get("n_estimators", 100), learning_rate=p.get("learning_rate", 0.1), max_depth=p.get("max_depth", 5), random_state=rs, n_jobs=-1)
            else:
                raise ValueError(f"Unsupported regression model: {model_name}")

    def fit_scaler(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fits standard scaler strictly on training inputs."""
        self.scaler.fit(X_train)
        return pd.DataFrame(self.scaler.transform(X_train), columns=X_train.columns)

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms evaluation features using fitted training scaler."""
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)

    def train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        scale_inputs: bool = True
    ) -> Any:
        """Fits a specific model architecture on training inputs."""
        model = self.instantiate_model(model_name, params=params)
        X_fit = self.fit_scaler(X_train) if scale_inputs else X_train
        model.fit(X_fit, y_train)
        self.trained_models[model_name] = model
        return model

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scale_inputs: bool = True
    ) -> Dict[str, Any]:
        """Trains every supported model in the registry for benchmark comparison."""
        names = self.get_supported_model_names()
        X_fit = self.fit_scaler(X_train) if scale_inputs else X_train

        for name in names:
            # For SVM, limit training size if dataset is huge (>5000) for interactive speed
            if name == "SVM" and len(X_fit) > 3000:
                sub_idx = np.random.RandomState(self.random_state).choice(len(X_fit), 3000, replace=False)
                m = self.instantiate_model(name)
                m.fit(X_fit.iloc[sub_idx], y_train.iloc[sub_idx])
            else:
                m = self.instantiate_model(name)
                m.fit(X_fit, y_train)
            self.trained_models[name] = m

        return self.trained_models

    def save_artifact(self, model_name: str, filepath: str):
        """Serializes trained model artifact to disk."""
        if model_name not in self.trained_models:
            raise KeyError(f"Model '{model_name}' has not been trained yet.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.trained_models[model_name], filepath)
