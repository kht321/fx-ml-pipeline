"""
Market Volatility Features Processor - Silver Layer

Repository Location: fx-ml-pipeline/src_clean/data_pipelines/silver/market_volatility_processor.py

Purpose:
    Processes bronze layer market data into volatility estimator features.
    Computes Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang estimators.

Input:
    - Bronze market data: data_clean/bronze/market/*.ndjson

Output:
    - Silver volatility features: data_clean/silver/market/volatility/*.csv
    - Features: 10 volatility estimators per candle

Usage:
    python src_clean/data_pipelines/silver/market_volatility_processor.py \
        --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
        --output data_clean/silver/market/volatility/spx500_volatility.csv
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolatilityProcessor:
    """Computes advanced volatility estimators from OHLC data."""

    def __init__(self, input_path: Path, output_path: Path, window: int = 20):
        self.input_path = input_path
        self.output_path = output_path
        self.window = window

    def load_candles(self) -> pd.DataFrame:
        """Load candles from bronze layer."""
        logger.info(f"Loading candles from {self.input_path}")

        candles = []
        with open(self.input_path, 'r') as f:
            for line in f:
                try:
                    candles.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values('time').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} candles")
        return df

    def garman_klass(self, df: pd.DataFrame) -> pd.Series:
        """
        Garman-Klass volatility estimator.

        GK = sqrt((1/n) * sum(0.5 * (log(H/L))^2 - (2*log(2)-1) * (log(C/O))^2))

        More efficient than close-to-close, uses OHLC information.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        open_price = df['open']

        term1 = 0.5 * (np.log(high / low)) ** 2
        term2 = (2 * np.log(2) - 1) * (np.log(close / open_price)) ** 2

        gk = np.sqrt((term1 - term2).rolling(window=self.window).mean())

        return gk

    def parkinson(self, df: pd.DataFrame) -> pd.Series:
        """
        Parkinson volatility estimator.

        P = sqrt((1/(4*n*log(2))) * sum((log(H/L))^2))

        Uses only high and low prices.
        """
        high = df['high']
        low = df['low']

        parkinson = np.sqrt(
            (1 / (4 * np.log(2))) *
            ((np.log(high / low)) ** 2).rolling(window=self.window).mean()
        )

        return parkinson

    def rogers_satchell(self, df: pd.DataFrame) -> pd.Series:
        """
        Rogers-Satchell volatility estimator.

        RS = sqrt((1/n) * sum(log(H/C) * log(H/O) + log(L/C) * log(L/O)))

        Allows for drift, doesn't assume zero mean returns.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        open_price = df['open']

        rs = np.sqrt(
            (
                np.log(high / close) * np.log(high / open_price) +
                np.log(low / close) * np.log(low / open_price)
            ).rolling(window=self.window).mean()
        )

        return rs

    def yang_zhang(self, df: pd.DataFrame) -> pd.Series:
        """
        Yang-Zhang volatility estimator.

        YZ = sqrt(open_vol + k*close_vol + (1-k)*rs_vol)

        Combines overnight, open-to-close, and Rogers-Satchell volatility.
        Most accurate estimator but complex.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        open_price = df['open']

        # Overnight volatility
        log_oc = np.log(open_price / close.shift(1))
        overnight_vol = log_oc.rolling(window=self.window).var()

        # Open-to-close volatility
        log_co = np.log(close / open_price)
        open_close_vol = log_co.rolling(window=self.window).var()

        # Rogers-Satchell component
        rs_component = (
            np.log(high / close) * np.log(high / open_price) +
            np.log(low / close) * np.log(low / open_price)
        ).rolling(window=self.window).mean()

        # Weighting factor
        k = 0.34 / (1.34 + (self.window + 1) / (self.window - 1))

        # Yang-Zhang estimator
        yz = np.sqrt(overnight_vol + k * open_close_vol + (1 - k) * rs_component)

        return yz

    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all volatility features."""
        logger.info("Computing volatility features...")

        result = df[['time', 'instrument', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Historical volatility (close-to-close)
        returns = df['close'].pct_change()
        result['hist_vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        result['hist_vol_50'] = returns.rolling(window=50).std() * np.sqrt(252)

        # Advanced estimators
        result['gk_vol'] = self.garman_klass(df) * np.sqrt(252)  # Annualized
        result['parkinson_vol'] = self.parkinson(df) * np.sqrt(252)
        result['rs_vol'] = self.rogers_satchell(df) * np.sqrt(252)
        result['yz_vol'] = self.yang_zhang(df) * np.sqrt(252)

        # Volatility of volatility
        result['vol_of_vol'] = result['hist_vol_20'].rolling(window=20).std()

        # Volatility regimes
        vol_median = result['hist_vol_20'].median()
        result['vol_regime_low'] = (result['hist_vol_20'] < vol_median * 0.7).astype(int)
        result['vol_regime_high'] = (result['hist_vol_20'] > vol_median * 1.5).astype(int)

        # Realized range-based volatility
        result['realized_range'] = (df['high'] - df['low']) / df['close']
        result['realized_range_ma'] = result['realized_range'].rolling(window=20).mean()

        # Exponentially weighted volatility
        result['ewma_vol'] = returns.ewm(span=20).std() * np.sqrt(252)

        logger.info(f"Computed {len(result.columns) - 7} volatility features")
        return result

    def save_features(self, df: pd.DataFrame):
        """Save features to silver layer."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        threshold = len(df.columns) * 0.5
        initial_rows = len(df)
        df = df.dropna(thresh=threshold)
        dropped = initial_rows - len(df)

        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with excessive missing values")

        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {self.output_path}")

    def run(self):
        """Execute processing pipeline."""
        logger.info("="*80)
        logger.info("Volatility Features Processor - Silver Layer")
        logger.info("="*80)

        df = self.load_candles()
        if df.empty:
            logger.error("No data to process")
            return

        features_df = self.compute_volatility_features(df)
        self.save_features(features_df)

        logger.info("="*80)
        logger.info("Volatility features processing complete")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20, help="Rolling window size")
    args = parser.parse_args()

    processor = VolatilityProcessor(args.input, args.output, args.window)
    processor.run()


if __name__ == "__main__":
    main()
