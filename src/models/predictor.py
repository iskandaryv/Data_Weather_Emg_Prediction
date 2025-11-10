"""
Machine learning models for weather emergency prediction.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from ..utils.logger import setup_logger
from ..utils.config import MODEL_DIR, MODEL_CONFIG

logger = setup_logger(__name__)


class EmergencyPredictor:
    """ML model for predicting weather emergencies."""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize predictor.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.feature_names = None
        self.is_trained = False

    def _create_model(self, model_type: str):
        """Create ML model based on type."""
        random_state = MODEL_CONFIG['random_state']

        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=random_state
            )
        elif model_type == 'logistic':
            return LogisticRegression(
                random_state=random_state,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation: bool = True
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X: Feature DataFrame
            y: Target Series
            validation: Whether to perform validation

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Training {self.model_type} model")

        self.feature_names = X.columns.tolist()

        if validation:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=MODEL_CONFIG['validation_size'],
                random_state=MODEL_CONFIG['random_state'],
                stratify=y if len(np.unique(y)) > 1 else None
            )

            self.model.fit(X_train, y_train)

            # Validation metrics
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1] if len(np.unique(y)) > 1 else y_pred

            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
            }

            if len(np.unique(y)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)

            logger.info(f"Validation metrics: {metrics}")
        else:
            self.model.fit(X, y)
            metrics = {}

        self.is_trained = True
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Probability array
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return predictions as probabilities
            preds = self.model.predict(X)
            return np.column_stack([1 - preds, preds])

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.

        Returns:
            DataFrame with feature importances
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df

    def save(self, filename: str):
        """
        Save model to disk.

        Args:
            filename: Filename for saved model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        filepath = MODEL_DIR / filename

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filename: str) -> 'EmergencyPredictor':
        """
        Load model from disk.

        Args:
            filename: Filename of saved model

        Returns:
            Loaded EmergencyPredictor instance
        """
        filepath = MODEL_DIR / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']
        predictor.is_trained = True

        logger.info(f"Model loaded from {filepath}")
        return predictor

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X: Feature DataFrame
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1] if len(np.unique(y)) > 1 else y_pred

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred)
        }

        if len(np.unique(y)) > 1:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)

        return metrics


class MultiEmergencyPredictor:
    """Multi-class predictor for different emergency types."""

    def __init__(self):
        self.models = {}
        self.emergency_types = []

    def train(
        self,
        X: pd.DataFrame,
        emergency_type: pd.Series,
        emergency_occurred: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Train separate models for each emergency type.

        Args:
            X: Feature DataFrame
            emergency_type: Series with emergency types
            emergency_occurred: Binary series indicating emergency

        Returns:
            Dictionary of metrics for each type
        """
        logger.info("Training multi-emergency predictor")

        self.emergency_types = emergency_type[emergency_occurred == 1].unique().tolist()
        all_metrics = {}

        for emg_type in self.emergency_types:
            logger.info(f"Training model for {emg_type}")

            # Create binary target for this emergency type
            y = (emergency_type == emg_type).astype(int)

            # Train model
            model = EmergencyPredictor(model_type='random_forest')
            metrics = model.train(X, y, validation=True)

            self.models[emg_type] = model
            all_metrics[emg_type] = metrics

        return all_metrics

    def predict(
        self,
        X: pd.DataFrame,
        return_probabilities: bool = False
    ) -> pd.DataFrame:
        """
        Predict emergency types and probabilities.

        Args:
            X: Feature DataFrame
            return_probabilities: Whether to return probabilities

        Returns:
            DataFrame with predictions for each type
        """
        predictions = pd.DataFrame(index=X.index)

        for emg_type, model in self.models.items():
            if return_probabilities:
                predictions[f'{emg_type}_prob'] = model.predict_proba(X)[:, 1]
            else:
                predictions[emg_type] = model.predict(X)

        return predictions

    def save(self, base_filename: str):
        """Save all models."""
        for emg_type, model in self.models.items():
            filename = f"{base_filename}_{emg_type}.pkl"
            model.save(filename)

        logger.info(f"Saved {len(self.models)} models")

    @classmethod
    def load(cls, base_filename: str, emergency_types: List[str]) -> 'MultiEmergencyPredictor':
        """Load all models."""
        predictor = cls()
        predictor.emergency_types = emergency_types

        for emg_type in emergency_types:
            filename = f"{base_filename}_{emg_type}.pkl"
            try:
                predictor.models[emg_type] = EmergencyPredictor.load(filename)
            except FileNotFoundError:
                logger.warning(f"Model for {emg_type} not found")

        logger.info(f"Loaded {len(predictor.models)} models")
        return predictor
