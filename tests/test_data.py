"""
Tests for data loading functionality.
"""
import pytest
import pandas as pd
from datetime import datetime

from src.data import WeatherDataLoader, EmergencyDataLoader


class TestWeatherDataLoader:
    """Test WeatherDataLoader class."""

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        loader = WeatherDataLoader()
        df = loader.generate_synthetic_data(start_year=2020, num_years=2, save=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'date' in df.columns
        assert 'temperature' in df.columns
        assert 'precipitation' in df.columns
        assert 'humidity' in df.columns
        assert 'wind_speed' in df.columns
        assert 'pressure' in df.columns

    def test_data_ranges(self):
        """Test that generated data is within expected ranges."""
        loader = WeatherDataLoader()
        df = loader.generate_synthetic_data(start_year=2020, num_years=1, save=False)

        assert df['humidity'].between(0, 100).all()
        assert df['precipitation'].min() >= 0
        assert df['wind_speed'].min() >= 0


class TestEmergencyDataLoader:
    """Test EmergencyDataLoader class."""

    def test_generate_synthetic_emergencies(self):
        """Test synthetic emergency generation."""
        # First generate weather data
        weather_loader = WeatherDataLoader()
        weather_df = weather_loader.generate_synthetic_data(start_year=2020, num_years=1, save=False)

        # Generate emergencies
        emg_loader = EmergencyDataLoader()
        emg_df = emg_loader.generate_synthetic_emergencies(weather_df, save=False)

        assert isinstance(emg_df, pd.DataFrame)

        if len(emg_df) > 0:
            assert 'date' in emg_df.columns
            assert 'type' in emg_df.columns
            assert 'severity' in emg_df.columns
            assert emg_df['severity'].between(0, 10).all()
