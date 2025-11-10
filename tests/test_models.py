"""
Tests for ML models.
"""
import pytest
import pandas as pd
import numpy as np

from src.models import EmergencyPredictor


class TestEmergencyPredictor:
    """Test EmergencyPredictor class."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000

        X = pd.DataFrame({
            'temperature': np.random.normal(15, 10, n_samples),
            'precipitation': np.random.gamma(2, 5, n_samples),
            'humidity': np.random.uniform(30, 80, n_samples),
            'wind_speed': np.random.gamma(3, 2, n_samples),
        })

        # Create target: emergency if temp > 35 or precipitation > 50
        y = ((X['temperature'] > 35) | (X['precipitation'] > 50)).astype(int)

        return X, y

    def test_model_initialization(self):
        """Test model initialization."""
        predictor = EmergencyPredictor(model_type='random_forest')
        assert predictor.model_type == 'random_forest'
        assert not predictor.is_trained

    def test_model_training(self, sample_training_data):
        """Test model training."""
        X, y = sample_training_data

        predictor = EmergencyPredictor(model_type='random_forest')
        metrics = predictor.train(X, y, validation=True)

        assert predictor.is_trained
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

    def test_model_prediction(self, sample_training_data):
        """Test model prediction."""
        X, y = sample_training_data

        predictor = EmergencyPredictor(model_type='random_forest')
        predictor.train(X, y, validation=False)

        predictions = predictor.predict(X.head(10))

        assert len(predictions) == 10
        assert predictions.dtype in [np.int32, np.int64]

    def test_model_predict_proba(self, sample_training_data):
        """Test probability prediction."""
        X, y = sample_training_data

        predictor = EmergencyPredictor(model_type='random_forest')
        predictor.train(X, y, validation=False)

        probas = predictor.predict_proba(X.head(10))

        assert probas.shape == (10, 2)
        assert np.all((probas >= 0) & (probas <= 1))
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_feature_importance(self, sample_training_data):
        """Test feature importance extraction."""
        X, y = sample_training_data

        predictor = EmergencyPredictor(model_type='random_forest')
        predictor.train(X, y, validation=False)

        importance_df = predictor.get_feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(X.columns)
