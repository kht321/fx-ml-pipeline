"""
Market Technical Features Processor - Silver Layer

Repository Location: fx-ml-pipeline/src_clean/data_pipelines/silver/market_technical_processor.py

Purpose:
    Processes bronze layer market data (OHLCV candles) into technical indicator features.
    Computes RSI, MACD, Bollinger Bands, Moving Averages, ATR, ADX, and momentum indicators.

Input:
    - Bronze market data: data_clean/bronze/market/*.ndjson

Output:
    - Silver technical features: data_clean/silver/market/technical/*.csv
    - Features: 17 technical indicators per candle

Usage:
    python src_clean/data_pipelines/silver/market_technical_processor.py \
        --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
        --output data_clean/silver/market/technical/spx500_technical.csv
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalFeaturesProcessor:
    """Computes technical indicators from OHLCV candle data."""

    def __init__(self, input_path: Path, output_path: Path, min_periods: int = 50):
        """
        Initialize processor.

        Parameters
        ----------
        input_path : Path
            Bronze NDJSON file with candles
        output_path : Path
            Silver CSV file for technical features
        min_periods : int
            Minimum periods required before computing indicators
        """
        self.input_path = input_path
        self.output_path = output_path
        self.min_periods = min_periods

    def load_candles(self) -> pd.DataFrame:
        """Load OHLCV candles from bronze layer."""
        logger.info(f"Loading candles from {self.input_path}")

        candles = []
        with open(self.input_path, 'r') as f:
            for line in f:
                try:
                    candles.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(candles)

        if df.empty:
            logger.error("No candles loaded")
            return df

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values('time').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} candles")
        logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def compute_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """
        Compute MACD (Moving Average Convergence Divergence).

        Returns
        -------
        macd : pd.Series
            MACD line (fast EMA - slow EMA)
        signal_line : pd.Series
            Signal line (EMA of MACD)
        histogram : pd.Series
            MACD histogram (MACD - signal)
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return macd, signal_line, histogram

    def compute_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """
        Compute Bollinger Bands.

        Returns
        -------
        upper : pd.Series
            Upper band (MA + std_dev * std)
        middle : pd.Series
            Middle band (Simple Moving Average)
        lower : pd.Series
            Lower band (MA - std_dev * std)
        bandwidth : pd.Series
            Band width (upper - lower) / middle
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        bandwidth = (upper - lower) / middle

        return upper, middle, lower, bandwidth

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Compute Average True Range (ATR).

        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = Moving average of True Range
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Compute Average Directional Index (ADX).

        Measures trend strength (0-100).
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Plus and Minus Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth with Wilder's smoothing
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Directional Index
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

        # ADX is smoothed DX
        adx = dx.rolling(window=period).mean()

        return adx

    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators."""
        logger.info("Computing technical features...")

        if len(df) < self.min_periods:
            logger.warning(f"Insufficient data: {len(df)} < {self.min_periods}")
            return pd.DataFrame()

        result = df[['time', 'instrument', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Price returns
        result['return_1'] = df['close'].pct_change(1)
        result['return_5'] = df['close'].pct_change(5)
        result['return_10'] = df['close'].pct_change(10)

        # RSI
        result['rsi_14'] = self.compute_rsi(df['close'], period=14)
        result['rsi_20'] = self.compute_rsi(df['close'], period=20)

        # MACD
        macd, signal, histogram = self.compute_macd(df['close'])
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_histogram'] = histogram

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width = self.compute_bollinger_bands(df['close'])
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower
        result['bb_width'] = bb_width
        result['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # Moving Averages
        result['sma_7'] = df['close'].rolling(window=7).mean()
        result['sma_14'] = df['close'].rolling(window=14).mean()
        result['sma_21'] = df['close'].rolling(window=21).mean()
        result['sma_50'] = df['close'].rolling(window=50).mean()

        result['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
        result['ema_14'] = df['close'].ewm(span=14, adjust=False).mean()
        result['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

        # ATR
        result['atr_14'] = self.compute_atr(df, period=14)

        # ADX
        result['adx_14'] = self.compute_adx(df, period=14)

        # Momentum
        result['momentum_5'] = df['close'] - df['close'].shift(5)
        result['momentum_10'] = df['close'] - df['close'].shift(10)
        result['momentum_20'] = df['close'] - df['close'].shift(20)

        # Rate of Change
        result['roc_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
        result['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100

        # Volatility
        result['volatility_20'] = result['return_1'].rolling(window=20).std()
        result['volatility_50'] = result['return_1'].rolling(window=50).std()

        logger.info(f"Computed {len(result.columns) - 7} technical features")

        return result

    def save_features(self, df: pd.DataFrame):
        """Save technical features to silver layer."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove rows with too many NaN values (first few rows while indicators warm up)
        threshold = len(df.columns) * 0.5
        initial_rows = len(df)
        df = df.dropna(thresh=threshold)
        dropped = initial_rows - len(df)

        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with excessive missing values")

        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {self.output_path}")

    def run(self):
        """Execute the full processing pipeline."""
        logger.info("="*80)
        logger.info("Technical Features Processor - Silver Layer")
        logger.info("="*80)

        # Load data
        df = self.load_candles()
        if df.empty:
            logger.error("No data to process")
            return

        # Compute features
        features_df = self.compute_technical_features(df)
        if features_df.empty:
            logger.error("No features computed")
            return

        # Save to silver layer
        self.save_features(features_df)

        logger.info("="*80)
        logger.info("Technical features processing complete")
        logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Bronze NDJSON file with candles"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Silver CSV file for technical features"
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=50,
        help="Minimum periods before computing indicators"
    )

    args = parser.parse_args()

    processor = TechnicalFeaturesProcessor(
        input_path=args.input,
        output_path=args.output,
        min_periods=args.min_periods
    )

    processor.run()


if __name__ == "__main__":
    main()
