"""
Data preprocessing and feature engineering for weather emergency prediction.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.interpolate import interp1d
from datetime import timedelta

from ..utils.logger import setup_logger
from ..utils.config import PROCESSED_DATA_DIR, WEATHER_FEATURES

logger = setup_logger(__name__)


class DataPreprocessor:
    """Preprocess weather and emergency data for modeling."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.fitted = False

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers using z-score method.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            n_std: Number of standard deviations for outlier threshold

        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers from {len(columns)} columns")

        df_clean = df.copy()

        for col in columns:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                mask = np.abs(df_clean[col] - mean) <= n_std * std
                df_clean = df_clean[mask]

        logger.info(f"Removed {len(df) - len(df_clean)} outlier rows")
        return df_clean

    def interpolate_missing_data(
        self,
        df: pd.DataFrame,
        method: str = 'linear'
    ) -> pd.DataFrame:
        """
        Interpolate missing data points.

        Args:
            df: Input DataFrame
            method: Interpolation method ('linear', 'cubic', etc.)

        Returns:
            DataFrame with interpolated values
        """
        logger.info(f"Interpolating missing data using {method} method")

        df_interp = df.copy()

        # Interpolate numeric columns
        numeric_cols = df_interp.select_dtypes(include=[np.number]).columns
        df_interp[numeric_cols] = df_interp[numeric_cols].interpolate(
            method=method,
            limit_direction='both'
        )

        return df_interp

    def create_time_features(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Create time-based features from date column.

        Args:
            df: Input DataFrame
            date_col: Name of date column

        Returns:
            DataFrame with additional time features
        """
        logger.info("Creating time-based features")

        df_time = df.copy()

        if date_col in df_time.columns:
            df_time[date_col] = pd.to_datetime(df_time[date_col])

            df_time['year'] = df_time[date_col].dt.year
            df_time['month'] = df_time[date_col].dt.month
            df_time['day'] = df_time[date_col].dt.day
            df_time['day_of_year'] = df_time[date_col].dt.dayofyear
            df_time['week_of_year'] = df_time[date_col].dt.isocalendar().week
            df_time['season'] = df_time['month'] % 12 // 3 + 1

            # Cyclical encoding for month
            df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
            df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)

            # Cyclical encoding for day of year
            df_time['day_sin'] = np.sin(2 * np.pi * df_time['day_of_year'] / 365)
            df_time['day_cos'] = np.cos(2 * np.pi * df_time['day_of_year'] / 365)

        return df_time

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [7, 14, 30],
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.

        Args:
            df: Input DataFrame (must be sorted by date)
            windows: List of window sizes in days
            features: List of features to compute rolling stats

        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features with windows: {windows}")

        df_roll = df.copy()

        if features is None:
            features = WEATHER_FEATURES

        for feature in features:
            if feature in df_roll.columns:
                for window in windows:
                    # Rolling mean
                    df_roll[f'{feature}_rolling_mean_{window}d'] = \
                        df_roll[feature].rolling(window=window, min_periods=1).mean()

                    # Rolling std
                    df_roll[f'{feature}_rolling_std_{window}d'] = \
                        df_roll[feature].rolling(window=window, min_periods=1).std()

                    # Rolling min/max
                    df_roll[f'{feature}_rolling_min_{window}d'] = \
                        df_roll[feature].rolling(window=window, min_periods=1).min()

                    df_roll[f'{feature}_rolling_max_{window}d'] = \
                        df_roll[feature].rolling(window=window, min_periods=1).max()

        return df_roll

    def create_lag_features(
        self,
        df: pd.DataFrame,
        lags: List[int] = [1, 3, 7],
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create lagged features.

        Args:
            df: Input DataFrame
            lags: List of lag values in days
            features: List of features to create lags

        Returns:
            DataFrame with lag features
        """
        logger.info(f"Creating lag features: {lags}")

        df_lag = df.copy()

        if features is None:
            features = WEATHER_FEATURES

        for feature in features:
            if feature in df_lag.columns:
                for lag in lags:
                    df_lag[f'{feature}_lag_{lag}d'] = df_lag[feature].shift(lag)

        return df_lag

    def aggregate_by_period(
        self,
        df: pd.DataFrame,
        period: str = 'M',
        agg_funcs: List[str] = ['mean', 'min', 'max', 'std']
    ) -> pd.DataFrame:
        """
        Aggregate data by time period.

        Args:
            df: Input DataFrame
            period: Resampling period ('D', 'W', 'M', 'Y')
            agg_funcs: Aggregation functions

        Returns:
            Aggregated DataFrame
        """
        logger.info(f"Aggregating data by {period}")

        df_agg = df.copy()

        if 'date' in df_agg.columns:
            df_agg = df_agg.set_index('date')

            numeric_cols = df_agg.select_dtypes(include=[np.number]).columns

            df_agg = df_agg[numeric_cols].resample(period).agg(agg_funcs)
            df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
            df_agg = df_agg.reset_index()

        return df_agg

    def merge_weather_emergency(
        self,
        weather_df: pd.DataFrame,
        emergency_df: pd.DataFrame,
        window_days: int = 1
    ) -> pd.DataFrame:
        """
        Merge weather and emergency data with time window.

        Args:
            weather_df: Weather DataFrame
            emergency_df: Emergency DataFrame
            window_days: Days before emergency to consider

        Returns:
            Merged DataFrame
        """
        logger.info("Merging weather and emergency data")

        df = weather_df.copy()
        df['has_emergency'] = 0
        df['emergency_type'] = 'none'
        df['emergency_severity'] = 0.0

        if len(emergency_df) > 0:
            for _, emg in emergency_df.iterrows():
                emg_date = pd.to_datetime(emg['date'])

                # Mark days within window before emergency
                mask = (
                    (df['date'] >= emg_date - timedelta(days=window_days)) &
                    (df['date'] <= emg_date)
                )

                df.loc[mask, 'has_emergency'] = 1
                df.loc[mask, 'emergency_type'] = emg['type']
                df.loc[mask, 'emergency_severity'] = emg.get('severity', 5.0)

        return df

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'has_emergency',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.

        Args:
            df: Input DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns (if None, use all numeric)

        Returns:
            Tuple of (X, y) DataFrames
        """
        logger.info("Preparing training data")

        df_prep = df.copy()

        # Remove rows with NaN in target
        df_prep = df_prep.dropna(subset=[target_col])

        # Select features
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != target_col]

        # Remove date and identifier columns
        exclude_cols = ['date', 'latitude', 'longitude', target_col,
                       'emergency_type', 'emergency_severity']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]

        X = df_prep[feature_cols]
        y = df_prep[target_col]

        # Handle any remaining NaN values
        X = X.fillna(X.mean())

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")

        return X, y

    def scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using StandardScaler.

        Args:
            X: Feature DataFrame
            fit: Whether to fit scaler (True for train, False for test)

        Returns:
            Scaled DataFrame
        """
        logger.info(f"Scaling features (fit={fit})")

        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)

        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return X_scaled
