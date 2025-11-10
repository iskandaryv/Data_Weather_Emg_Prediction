"""
Tests for data preprocessing.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data import DataPreprocessor


class TestDataPreprocessor:
    """Test DataPreprocessor class."""

    @pytest.fixture
    def sample_weather_df(self):
        """Create sample weather data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        df = pd.DataFrame({
            'date': dates,
            'temperature': np.random.normal(15, 10, len(dates)),
            'precipitation': np.random.gamma(2, 5, len(dates)),
            'humidity': np.random.uniform(30, 80, len(dates)),
            'wind_speed': np.random.gamma(3, 2, len(dates)),
            'pressure': np.random.normal(1013, 10, len(dates))
        })
        return df

    def test_create_time_features(self, sample_weather_df):
        """Test time feature creation."""
        preprocessor = DataPreprocessor()
        df = preprocessor.create_time_features(sample_weather_df)

        assert 'year' in df.columns
        assert 'month' in df.columns
        assert 'day' in df.columns
        assert 'day_of_year' in df.columns
        assert 'season' in df.columns
        assert 'month_sin' in df.columns
        assert 'month_cos' in df.columns

    def test_remove_outliers(self, sample_weather_df):
        """Test outlier removal."""
        # Add some outliers
        df = sample_weather_df.copy()
        df.loc[0, 'temperature'] = 1000  # Extreme outlier

        preprocessor = DataPreprocessor()
        df_clean = preprocessor.remove_outliers(df, ['temperature'], n_std=3.0)

        assert len(df_clean) < len(df)

    def test_create_rolling_features(self, sample_weather_df):
        """Test rolling feature creation."""
        preprocessor = DataPreprocessor()
        df = preprocessor.create_rolling_features(
            sample_weather_df,
            windows=[7],
            features=['temperature']
        )

        assert 'temperature_rolling_mean_7d' in df.columns
        assert 'temperature_rolling_std_7d' in df.columns

    def test_create_lag_features(self, sample_weather_df):
        """Test lag feature creation."""
        preprocessor = DataPreprocessor()
        df = preprocessor.create_lag_features(
            sample_weather_df,
            lags=[1, 3],
            features=['temperature']
        )

        assert 'temperature_lag_1d' in df.columns
        assert 'temperature_lag_3d' in df.columns

    def test_scale_features(self, sample_weather_df):
        """Test feature scaling."""
        preprocessor = DataPreprocessor()

        X = sample_weather_df[['temperature', 'precipitation', 'humidity']]
        X_scaled = preprocessor.scale_features(X, fit=True)

        # Check that scaled features have mean ~0 and std ~1
        assert abs(X_scaled.mean().mean()) < 0.1
        assert abs(X_scaled.std().mean() - 1.0) < 0.1
