"""
Market Microstructure Features Processor - Silver Layer

Repository Location: fx-ml-pipeline/src_clean/data_pipelines/silver/market_microstructure_processor.py

Purpose:
    Processes bronze layer market data into microstructure features.
    Computes volume patterns, high-low range, order flow proxies, and liquidity metrics.

Input:
    - Bronze market data: data_clean/bronze/market/*.ndjson

Output:
    - Silver microstructure features: data_clean/silver/market/microstructure/*.csv
    - Features: 10 microstructure indicators per candle

Usage:
    python src_clean/data_pipelines/silver/market_microstructure_processor.py \
        --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
        --output data_clean/silver/market/microstructure/spx500_microstructure.csv
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


class MicrostructureProcessor:
    """Computes market microstructure features from OHLCV data."""

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path

    def load_candles(self) -> pd.DataFrame:
        """Load candles from bronze layer and forward-fill gaps at 1-minute granularity."""
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
        df = df.sort_values('time')

        if df.empty:
            logger.warning("No candle data found.")
            return df

        df = df.set_index('time')
        original_index = df.index
        full_index = pd.date_range(
            start=original_index.min(),
            end=original_index.max(),
            freq='1T',  # strict 1-minute cadence
            tz=original_index.tz
        )

        df_full = df.reindex(full_index)
        original_series = pd.Series(df_full.index.isin(original_index), index=df_full.index)

        df_full = df_full.ffill()
        logger.info("Reindex completed. Adding gap metadata.")

        minutes_since_last = []
        gap_count = 0
        for original in original_series:
            if original:
                gap_count = 0
            else:
                gap_count += 1
            minutes_since_last.append(gap_count)

        df_full['minutes_since_last_data'] = pd.Series(minutes_since_last, index=df_full.index).astype('Int64')

        is_original_int = original_series.astype(int)
        df_full['is_backfilled'] = (1 - is_original_int).astype(int)

        orig_shifted = is_original_int.shift(1, fill_value=0)
        cumsum_orig = orig_shifted.cumsum()
        count_prev_360 = cumsum_orig - cumsum_orig.shift(360, fill_value=0)
        count_prev_60 = cumsum_orig - cumsum_orig.shift(60, fill_value=0)
        count_prev_30 = cumsum_orig - cumsum_orig.shift(30, fill_value=0)
        count_prev_30_60 = count_prev_60 - count_prev_30
        count_prev_30_360 = count_prev_360 - count_prev_30
        df_full['has_original_prev_30_to_60min'] = (count_prev_30_60 > 0).astype(int)
        df_full['has_original_prev_30_to_360min'] = (count_prev_30_360 > 0).astype(int)

        df_full = df_full.reset_index().rename(columns={'index': 'time'})

        logger.info(f"Loaded {len(df_full)} candles after filling gaps (original: {len(df)}). Percentage of backfill: {(df_full['is_backfilled']==1).mean():.1%}. Percentage of orig 30-60min: {(df_full['has_original_prev_30_to_60min']==1).mean():.1%}. Percentage of 30-360min: {(df_full['has_original_prev_30_to_360min']==1).mean():.1%}")
        return df_full

    def compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute microstructure features."""
        logger.info("Computing microstructure features...")

        result = df[['time', 'instrument', 'open', 'high', 'low', 'close', 'volume']].copy()

        # High-Low Range (proxy for volatility and liquidity)
        result['hl_range'] = df['high'] - df['low']
        result['hl_range_pct'] = result['hl_range'] / df['close']
        result['hl_range_ma20'] = result['hl_range'].rolling(window=20).mean()

        # Spread proxy (no bid-ask for index, use HL as proxy)
        result['spread_proxy'] = result['hl_range']
        result['spread_pct'] = result['spread_proxy'] / df['close']

        # Volume analysis
        result['volume_ma20'] = df['volume'].rolling(window=20).mean()
        result['volume_ma50'] = df['volume'].rolling(window=50).mean()
        result['volume_ratio'] = df['volume'] / result['volume_ma20']
        result['volume_zscore'] = (
            (df['volume'] - result['volume_ma20']) /
            df['volume'].rolling(window=20).std()
        )

        # Price impact proxy (price move per unit volume)
        price_change = df['close'].diff().abs()
        result['price_impact'] = price_change / (df['volume'] + 1)  # +1 to avoid division by zero
        result['price_impact_ma20'] = result['price_impact'].rolling(window=20).mean()

        # Order flow imbalance proxy
        # Positive if close > open (buying pressure), negative otherwise
        result['order_flow_imbalance'] = (df['close'] - df['open']) / result['hl_range']
        result['order_flow_imbalance'] = result['order_flow_imbalance'].fillna(0)

        # Amihud illiquidity measure (|return| / volume)
        returns = df['close'].pct_change().abs()
        result['illiquidity'] = returns / (df['volume'] + 1)
        result['illiquidity_ma20'] = result['illiquidity'].rolling(window=20).mean()

        # Volume-weighted indicators
        result['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        result['close_vwap_ratio'] = df['close'] / result['vwap']

        # Trade intensity (volume velocity)
        result['volume_velocity'] = df['volume'].diff()
        result['volume_acceleration'] = result['volume_velocity'].diff()

        logger.info(f"Computed {len(result.columns) - 7} microstructure features")
        return result

    def save_features(self, df: pd.DataFrame):
        """Save features to silver layer."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove rows with excessive NaN
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
        logger.info("Microstructure Features Processor - Silver Layer")
        logger.info("="*80)

        df = self.load_candles()
        if df.empty:
            logger.error("No data to process")
            return

        features_df = self.compute_microstructure_features(df)
        self.save_features(features_df)

        logger.info("="*80)
        logger.info("Microstructure features processing complete")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    processor = MicrostructureProcessor(args.input, args.output)
    processor.run()


if __name__ == "__main__":
    main()
