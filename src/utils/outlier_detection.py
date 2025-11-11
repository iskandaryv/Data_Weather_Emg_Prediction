"""
Outlier Detection & Handling (Выбросы)
Multiple methods for robust outlier detection in weather/emergency data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


class OutlierDetector:
    """
    Comprehensive outlier detection with multiple methods.
    Handles weather data anomalies and emergency case outliers.
    """

    def __init__(self, contamination=0.05):
        """
        Initialize outlier detector.

        Args:
            contamination: Expected proportion of outliers (default 5%)
        """
        self.contamination = contamination
        self.outlier_mask = None
        self.methods_used = []

    def detect_zscore(self, df: pd.DataFrame, columns: list, threshold=3.0) -> pd.DataFrame:
        """
        Z-score method: |z| > threshold are outliers.
        Good for univariate outlier detection.

        Args:
            df: DataFrame with data
            columns: Columns to check
            threshold: Z-score threshold (default 3.0)

        Returns:
            DataFrame with outlier_zscore column (True = outlier)
        """
        print(f"\n🔍 Z-Score Outlier Detection (threshold={threshold})")

        df_result = df.copy()
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
                col_outliers = z_scores > threshold
                outlier_mask |= col_outliers

                n_outliers = col_outliers.sum()
                print(f"   {col}: {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")

        df_result['outlier_zscore'] = outlier_mask
        self.methods_used.append('zscore')

        print(f"   ✅ Total outliers: {outlier_mask.sum()} ({outlier_mask.sum()/len(df)*100:.2f}%)")
        return df_result

    def detect_iqr(self, df: pd.DataFrame, columns: list, k=1.5) -> pd.DataFrame:
        """
        IQR (Interquartile Range) method: Tukey's fences.
        Outliers are outside [Q1 - k*IQR, Q3 + k*IQR].

        Args:
            df: DataFrame with data
            columns: Columns to check
            k: IQR multiplier (default 1.5, use 3.0 for extreme outliers)

        Returns:
            DataFrame with outlier_iqr column
        """
        print(f"\n🔍 IQR Outlier Detection (k={k})")

        df_result = df.copy()
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - k * IQR
                upper_bound = Q3 + k * IQR

                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_mask |= col_outliers

                n_outliers = col_outliers.sum()
                print(f"   {col}: {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
                print(f"      Range: [{lower_bound:.2f}, {upper_bound:.2f}]")

        df_result['outlier_iqr'] = outlier_mask
        self.methods_used.append('iqr')

        print(f"   ✅ Total outliers: {outlier_mask.sum()} ({outlier_mask.sum()/len(df)*100:.2f}%)")
        return df_result

    def detect_isolation_forest(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Isolation Forest: ML-based multivariate outlier detection.
        Excellent for high-dimensional data and complex patterns.

        Args:
            df: DataFrame with data
            columns: Columns to use as features

        Returns:
            DataFrame with outlier_iforest column and anomaly_score
        """
        print(f"\n🔍 Isolation Forest Outlier Detection")

        df_result = df.copy()

        # Prepare data
        X = df[columns].fillna(df[columns].mean())

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )

        predictions = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.score_samples(X)

        # -1 = outlier, 1 = inlier
        outlier_mask = predictions == -1

        df_result['outlier_iforest'] = outlier_mask
        df_result['anomaly_score_iforest'] = anomaly_scores

        self.methods_used.append('isolation_forest')

        n_outliers = outlier_mask.sum()
        print(f"   ✅ Found {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
        print(f"   📊 Anomaly score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")

        return df_result

    def detect_elliptic_envelope(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Elliptic Envelope: Assumes Gaussian distribution.
        Good for weather data with normal distribution.

        Args:
            df: DataFrame with data
            columns: Columns to use as features

        Returns:
            DataFrame with outlier_elliptic column and mahalanobis_distance
        """
        print(f"\n🔍 Elliptic Envelope Outlier Detection")

        df_result = df.copy()

        # Prepare data
        X = df[columns].fillna(df[columns].mean())

        # Fit Elliptic Envelope
        try:
            elliptic = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )

            predictions = elliptic.fit_predict(X)
            mahalanobis = elliptic.mahalanobis(X)

            # -1 = outlier, 1 = inlier
            outlier_mask = predictions == -1

            df_result['outlier_elliptic'] = outlier_mask
            df_result['mahalanobis_distance'] = mahalanobis

            self.methods_used.append('elliptic_envelope')

            n_outliers = outlier_mask.sum()
            print(f"   ✅ Found {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
            print(f"   📊 Mahalanobis distance range: [{mahalanobis.min():.3f}, {mahalanobis.max():.3f}]")

        except Exception as e:
            print(f"   ⚠️ Elliptic Envelope failed: {e}")
            df_result['outlier_elliptic'] = False
            df_result['mahalanobis_distance'] = 0.0

        return df_result

    def detect_mad(self, df: pd.DataFrame, columns: list, threshold=3.5) -> pd.DataFrame:
        """
        MAD (Median Absolute Deviation): Robust to extreme outliers.
        Better than z-score when data has many outliers.

        Args:
            df: DataFrame with data
            columns: Columns to check
            threshold: MAD threshold (default 3.5)

        Returns:
            DataFrame with outlier_mad column
        """
        print(f"\n🔍 MAD Outlier Detection (threshold={threshold})")

        df_result = df.copy()
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in columns:
            if col in df.columns:
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))

                # Modified z-score
                if mad != 0:
                    modified_z_score = 0.6745 * (df[col] - median) / mad
                    col_outliers = np.abs(modified_z_score) > threshold
                else:
                    col_outliers = pd.Series([False] * len(df))

                outlier_mask |= col_outliers

                n_outliers = col_outliers.sum()
                print(f"   {col}: {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")

        df_result['outlier_mad'] = outlier_mask
        self.methods_used.append('mad')

        print(f"   ✅ Total outliers: {outlier_mask.sum()} ({outlier_mask.sum()/len(df)*100:.2f}%)")
        return df_result

    def ensemble_detection(self, df: pd.DataFrame, columns: list,
                          methods=['zscore', 'iqr', 'isolation_forest'],
                          voting='majority') -> pd.DataFrame:
        """
        Ensemble method: Combine multiple detection methods.

        Args:
            df: DataFrame with data
            columns: Columns to check
            methods: List of methods to use
            voting: 'majority' (2+ methods) or 'unanimous' (all methods)

        Returns:
            DataFrame with outlier_ensemble column
        """
        print(f"\n🔍 Ensemble Outlier Detection")
        print(f"   Methods: {', '.join(methods)}")
        print(f"   Voting: {voting}")

        df_result = df.copy()

        # Run all methods
        if 'zscore' in methods:
            df_result = self.detect_zscore(df_result, columns)

        if 'iqr' in methods:
            df_result = self.detect_iqr(df_result, columns)

        if 'isolation_forest' in methods:
            df_result = self.detect_isolation_forest(df_result, columns)

        if 'elliptic' in methods:
            df_result = self.detect_elliptic_envelope(df_result, columns)

        if 'mad' in methods:
            df_result = self.detect_mad(df_result, columns)

        # Combine results
        outlier_cols = [col for col in df_result.columns if col.startswith('outlier_')]

        if voting == 'majority':
            # At least 50% of methods agree
            votes = df_result[outlier_cols].sum(axis=1)
            threshold = len(outlier_cols) / 2
            df_result['outlier_ensemble'] = votes >= threshold
        else:  # unanimous
            # All methods must agree
            df_result['outlier_ensemble'] = df_result[outlier_cols].all(axis=1)

        n_outliers = df_result['outlier_ensemble'].sum()
        print(f"\n   🎯 Ensemble result: {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")

        return df_result

    def handle_outliers(self, df: pd.DataFrame, method='cap',
                       outlier_col='outlier_ensemble') -> pd.DataFrame:
        """
        Handle detected outliers using various strategies.

        Args:
            df: DataFrame with outlier detection results
            method: 'remove', 'cap' (winsorization), 'interpolate', or 'flag'
            outlier_col: Column indicating outliers

        Returns:
            DataFrame with handled outliers
        """
        print(f"\n🛠️ Handling Outliers: {method}")

        df_result = df.copy()
        n_outliers = df_result[outlier_col].sum()

        if method == 'remove':
            # Remove outlier rows
            df_result = df_result[~df_result[outlier_col]]
            print(f"   ✅ Removed {n_outliers} outlier rows")
            print(f"   📊 Dataset size: {len(df)} → {len(df_result)}")

        elif method == 'cap':
            # Winsorization: cap at 1st and 99th percentiles
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if col not in ['outlier_ensemble', 'anomaly_score_iforest', 'mahalanobis_distance']:
                    q1 = df_result[col].quantile(0.01)
                    q99 = df_result[col].quantile(0.99)

                    df_result.loc[df_result[outlier_col], col] = df_result.loc[
                        df_result[outlier_col], col
                    ].clip(lower=q1, upper=q99)

            print(f"   ✅ Capped {n_outliers} outliers to 1-99 percentile range")

        elif method == 'interpolate':
            # Interpolate outlier values
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                if col not in ['outlier_ensemble', 'anomaly_score_iforest', 'mahalanobis_distance']:
                    # Set outliers to NaN and interpolate
                    df_result.loc[df_result[outlier_col], col] = np.nan
                    df_result[col] = df_result[col].interpolate(method='linear')

            print(f"   ✅ Interpolated {n_outliers} outliers")

        elif method == 'flag':
            # Just flag, don't modify
            print(f"   ✅ Flagged {n_outliers} outliers (no modification)")

        return df_result

    def generate_outlier_report(self, df: pd.DataFrame) -> dict:
        """
        Generate comprehensive outlier detection report.

        Args:
            df: DataFrame with outlier detection results

        Returns:
            Dictionary with outlier statistics
        """
        print("\n📋 Outlier Detection Report")
        print("=" * 70)

        report = {
            'total_records': len(df),
            'methods_used': self.methods_used,
            'outliers_by_method': {}
        }

        outlier_cols = [col for col in df.columns if col.startswith('outlier_')]

        for col in outlier_cols:
            method_name = col.replace('outlier_', '')
            n_outliers = df[col].sum()
            pct_outliers = n_outliers / len(df) * 100

            report['outliers_by_method'][method_name] = {
                'count': int(n_outliers),
                'percentage': round(pct_outliers, 2)
            }

            print(f"{method_name:20s}: {n_outliers:5d} ({pct_outliers:5.2f}%)")

        print("=" * 70)

        return report


def quick_outlier_detection(df: pd.DataFrame, columns: list,
                            method='ensemble') -> pd.DataFrame:
    """
    Quick outlier detection with recommended defaults.

    Args:
        df: DataFrame with data
        columns: Columns to check for outliers
        method: 'zscore', 'iqr', 'isolation_forest', or 'ensemble'

    Returns:
        DataFrame with outliers detected and handled
    """
    detector = OutlierDetector(contamination=0.05)

    if method == 'ensemble':
        df_result = detector.ensemble_detection(
            df, columns,
            methods=['zscore', 'iqr', 'isolation_forest'],
            voting='majority'
        )
    elif method == 'zscore':
        df_result = detector.detect_zscore(df, columns, threshold=3.0)
    elif method == 'iqr':
        df_result = detector.detect_iqr(df, columns, k=1.5)
    elif method == 'isolation_forest':
        df_result = detector.detect_isolation_forest(df, columns)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Generate report
    detector.generate_outlier_report(df_result)

    return df_result


if __name__ == "__main__":
    # Example usage
    print("Outlier Detection Module - Ready")
    print("Available methods:")
    print("  • Z-Score (univariate)")
    print("  • IQR / Tukey's Fences (univariate)")
    print("  • Isolation Forest (multivariate, ML)")
    print("  • Elliptic Envelope (multivariate, Gaussian)")
    print("  • MAD - Median Absolute Deviation (robust)")
    print("  • Ensemble (combines multiple methods)")
