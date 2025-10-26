"""
Advanced Feature Engineering for S&P 500 Prediction

Implements market microstructure, cross-asset, and time-based features
to improve model performance.

Repository Location: fx-ml-pipeline/src_clean/features/advanced_feature_engineering.py
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Create advanced features for financial prediction."""

    def __init__(self):
        """Initialize feature engineer."""
        self.features_created = []

    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: open, high, low, close, volume

        Returns
        -------
        pd.DataFrame
            DataFrame with additional features
        """
        logger.info("Adding market microstructure features...")

        # Order flow imbalance (if we had bid/ask volume)
        # Approximation using price and volume
        df['price_weighted_volume'] = df['close'] * df['volume']
        df['volume_imbalance'] = df['volume'].diff() / df['volume'].rolling(20).mean()

        # Distance from key price levels
        df['distance_from_vwap'] = (df['close'] - df['vwap']) / df['vwap'] if 'vwap' in df else 0
        df['distance_from_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['distance_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

        # Relative position in range
        df['position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)

        # Volume patterns
        df['volume_ratio_5_20'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        df['abnormal_volume'] = df['volume'] / df['volume'].rolling(50).mean() - 1

        # Price efficiency
        df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.0001)

        # Microstructure volatility
        df['high_low_ratio'] = df['high'] / df['low'] - 1
        df['close_to_close_vol'] = df['close'].pct_change().rolling(20).std()

        self.features_created.extend([
            'price_weighted_volume', 'volume_imbalance', 'distance_from_high_20',
            'distance_from_low_20', 'position_in_range', 'volume_ratio_5_20',
            'abnormal_volume', 'price_efficiency', 'high_low_ratio', 'close_to_close_vol'
        ])

        logger.info(f"Added {len(self.features_created)} microstructure features")
        return df

    def add_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'time' column

        Returns
        -------
        pd.DataFrame
            DataFrame with time features
        """
        logger.info("Adding time-based features...")

        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

            # Hour and minute
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute

            # Trading session indicators (in UTC for US markets)
            df['is_us_premarket'] = df['hour'].between(8, 14)  # 4am-9:30am ET
            df['is_us_regular'] = df['hour'].between(14, 20)   # 9:30am-4pm ET
            df['is_us_afterhours'] = df['hour'].between(20, 24) | df['hour'].between(0, 1)  # 4pm-8pm ET

            # Day of week
            df['day_of_week'] = df['time'].dt.dayofweek
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)

            # Month and quarter
            df['month'] = df['time'].dt.month
            df['quarter'] = df['time'].dt.quarter

            # Special times
            df['is_market_open'] = ((df['hour'] == 14) & (df['minute'] < 31)).astype(int)
            df['is_market_close'] = ((df['hour'] == 20) & (df['minute'] >= 30)).astype(int)

            # Cyclical encoding for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            # Cyclical encoding for day of week
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            self.features_created.extend([
                'hour', 'minute', 'is_us_premarket', 'is_us_regular', 'is_us_afterhours',
                'day_of_week', 'is_monday', 'is_friday', 'month', 'quarter',
                'is_market_open', 'is_market_close', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
            ])

        logger.info(f"Added {16} time-based features")
        return df

    def add_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime detection features.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain price and volume data

        Returns
        -------
        pd.DataFrame
            DataFrame with regime features
        """
        logger.info("Adding regime detection features...")

        # Trend regime
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        df['is_uptrend'] = (df['sma_20'] > df['sma_50']).astype(int)

        # Volatility regime
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        df['volatility_50'] = df['close'].pct_change().rolling(50).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
        df['high_volatility'] = (df['volatility_ratio'] > 1.5).astype(int)

        # Volume regime
        df['volume_20'] = df['volume'].rolling(20).mean()
        df['volume_50'] = df['volume'].rolling(50).mean()
        df['volume_regime'] = df['volume_20'] / df['volume_50']
        df['high_volume'] = (df['volume_regime'] > 1.2).astype(int)

        # Market stress indicator
        df['market_stress'] = (
            df['high_volatility'] * 0.5 +
            df['high_volume'] * 0.3 +
            (df['volatility_20'] > df['volatility_20'].rolling(252).mean() +
             2 * df['volatility_20'].rolling(252).std()).astype(int) * 0.2
        )

        self.features_created.extend([
            'sma_20', 'sma_50', 'trend_strength', 'is_uptrend',
            'volatility_20', 'volatility_50', 'volatility_ratio', 'high_volatility',
            'volume_20', 'volume_50', 'volume_regime', 'high_volume', 'market_stress'
        ])

        logger.info(f"Added {13} regime detection features")
        return df

    def add_pattern_recognition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add simple pattern recognition features.

        Parameters
        ----------
        df : pd.DataFrame
            OHLC data

        Returns
        -------
        pd.DataFrame
            DataFrame with pattern features
        """
        logger.info("Adding pattern recognition features...")

        # Candlestick patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']

        # Doji pattern (small body)
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)

        # Hammer pattern (small body, long lower shadow)
        df['is_hammer'] = ((df['body_size'] < 0.002) &
                           (df['lower_shadow'] > df['body_size'] * 2)).astype(int)

        # Engulfing pattern
        df['prev_body'] = df['body_size'].shift(1)
        df['is_engulfing'] = (df['body_size'] > df['prev_body'] * 1.5).astype(int)

        # Support/Resistance levels
        df['near_20d_high'] = (df['close'] > df['high'].rolling(20).max() * 0.98).astype(int)
        df['near_20d_low'] = (df['close'] < df['low'].rolling(20).min() * 1.02).astype(int)

        # Breakout detection
        df['breakout_up'] = (df['close'] > df['high'].rolling(20).max()).astype(int)
        df['breakout_down'] = (df['close'] < df['low'].rolling(20).min()).astype(int)

        self.features_created.extend([
            'body_size', 'upper_shadow', 'lower_shadow', 'is_doji', 'is_hammer',
            'is_engulfing', 'near_20d_high', 'near_20d_low', 'breakout_up', 'breakout_down'
        ])

        logger.info(f"Added {10} pattern recognition features")
        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all advanced features.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with OHLCV and time data

        Returns
        -------
        pd.DataFrame
            DataFrame with all additional features
        """
        logger.info("\n" + "="*80)
        logger.info("ADVANCED FEATURE ENGINEERING")
        logger.info("="*80)

        initial_features = len(df.columns)

        # Add all feature sets
        df = self.add_market_microstructure_features(df)
        df = self.add_time_based_features(df)
        df = self.add_regime_detection_features(df)
        df = self.add_pattern_recognition_features(df)

        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)

        total_features = len(df.columns) - initial_features

        logger.info("="*80)
        logger.info(f"FEATURE ENGINEERING COMPLETE")
        logger.info(f"Added {total_features} new features")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info("="*80 + "\n")

        return df


def main():
    """Example usage."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Add advanced features to market data")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV with market data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV with enhanced features"
    )

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Add features
    engineer = AdvancedFeatureEngineer()
    df_enhanced = engineer.add_all_features(df)

    # Save
    df_enhanced.to_csv(args.output, index=False)
    logger.info(f"Saved enhanced features to {args.output}")


if __name__ == "__main__":
    main()