"""
Data loader for weather and emergency data.
Handles data ingestion, validation, and preprocessing for Rostov region.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta

from ..utils.logger import setup_logger
from ..utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_GEO_LOCATION,
    START_YEAR,
    YEARS_OF_DATA,
    WEATHER_FEATURES
)

logger = setup_logger(__name__)


class WeatherDataLoader:
    """Load and validate weather data for specified geographic region."""

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = data_dir
        self.geo_location = DEFAULT_GEO_LOCATION

    def generate_synthetic_data(
        self,
        start_year: int = START_YEAR,
        num_years: int = YEARS_OF_DATA,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic weather data for demonstration.
        In production, this would load real data from APIs or files.

        Args:
            start_year: Starting year for data
            num_years: Number of years of data
            save: Whether to save generated data

        Returns:
            DataFrame with weather data
        """
        logger.info(f"Generating synthetic weather data for {num_years} years")

        # Generate date range
        start_date = datetime(start_year, 1, 1)
        end_date = start_date + timedelta(days=365 * num_years)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate synthetic weather data
        np.random.seed(42)
        n = len(dates)

        # Temperature with seasonal pattern
        day_of_year = dates.dayofyear
        temp_base = 10 + 15 * np.sin(2 * np.pi * day_of_year / 365)
        temperature = temp_base + np.random.normal(0, 5, n)

        # Precipitation with seasonal variation
        precip_prob = 0.3 + 0.2 * np.sin(2 * np.pi * day_of_year / 365 + np.pi/2)
        precipitation = np.random.gamma(2, 5, n) * (np.random.random(n) < precip_prob)

        # Humidity
        humidity = np.clip(
            50 + 20 * np.sin(2 * np.pi * day_of_year / 365 + np.pi/2) +
            np.random.normal(0, 10, n),
            0, 100
        )

        # Wind speed
        wind_speed = np.abs(np.random.gamma(3, 2, n))

        # Pressure
        pressure = 1013 + np.random.normal(0, 10, n)

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'latitude': self.geo_location['lat'],
            'longitude': self.geo_location['lon'],
            'temperature': temperature,
            'precipitation': precipitation,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure
        })

        if save:
            output_path = self.data_dir / 'weather_geodata.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved weather data to {output_path}")

        return df

    def load_weather_data(
        self,
        file_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load weather data from CSV file.

        Args:
            file_path: Path to CSV file (optional)

        Returns:
            DataFrame with weather data
        """
        if file_path is None:
            file_path = self.data_dir / 'weather_geodata.csv'

        if not file_path.exists():
            logger.warning(f"File {file_path} not found. Generating synthetic data.")
            return self.generate_synthetic_data()

        logger.info(f"Loading weather data from {file_path}")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        return df


class EmergencyDataLoader:
    """Load and validate emergency event data."""

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = data_dir

    def generate_synthetic_emergencies(
        self,
        weather_df: pd.DataFrame,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic emergency events based on weather conditions.

        Args:
            weather_df: Weather data DataFrame
            save: Whether to save generated data

        Returns:
            DataFrame with emergency events
        """
        logger.info("Generating synthetic emergency events")

        np.random.seed(42)

        # Define emergency conditions
        emergencies = []

        for idx, row in weather_df.iterrows():
            # Heatwave: temp > 35°C for 3+ consecutive days
            if row['temperature'] > 35 and np.random.random() < 0.3:
                emergencies.append({
                    'date': row['date'],
                    'type': 'heatwave',
                    'severity': min(10, (row['temperature'] - 35) / 2),
                    'latitude': row['latitude'],
                    'longitude': row['longitude']
                })

            # Drought: low precipitation over time
            if row['precipitation'] < 1 and row['humidity'] < 30 and np.random.random() < 0.1:
                emergencies.append({
                    'date': row['date'],
                    'type': 'drought',
                    'severity': np.random.uniform(3, 7),
                    'latitude': row['latitude'],
                    'longitude': row['longitude']
                })

            # Flood: heavy precipitation
            if row['precipitation'] > 50 and np.random.random() < 0.4:
                emergencies.append({
                    'date': row['date'],
                    'type': 'flood',
                    'severity': min(10, row['precipitation'] / 10),
                    'latitude': row['latitude'],
                    'longitude': row['longitude']
                })

            # Frost: very low temperature
            if row['temperature'] < -20 and np.random.random() < 0.3:
                emergencies.append({
                    'date': row['date'],
                    'type': 'frost',
                    'severity': min(10, abs(row['temperature'] + 20) / 2),
                    'latitude': row['latitude'],
                    'longitude': row['longitude']
                })

        df = pd.DataFrame(emergencies)

        if save and len(df) > 0:
            output_path = self.data_dir / 'emergencies_geodata.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} emergency events to {output_path}")

        return df

    def load_emergency_data(
        self,
        file_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load emergency data from CSV file.

        Args:
            file_path: Path to CSV file (optional)

        Returns:
            DataFrame with emergency events
        """
        if file_path is None:
            file_path = self.data_dir / 'emergencies_geodata.csv'

        if not file_path.exists():
            logger.warning(f"File {file_path} not found.")
            return pd.DataFrame()

        logger.info(f"Loading emergency data from {file_path}")
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        return df
