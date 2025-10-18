"""Transform historical OHLC candle data into Silver-level technical features for training.

This script processes historical candle data (hourly, daily, etc.) into technical features
for model training. For live tick streaming, use build_market_features.py instead.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np


def parse_args(argv: Iterable[str] = None) -> argparse.Namespace:
    """Define the command-line interface and parse user inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to NDJSON file with OHLC candle data",
    )
    parser.add_argument(
        "--output-technical",
        type=Path,
        default=Path("data/market/silver/technical_features/sgd_vs_majors.csv"),
        help="CSV destination for technical features",
    )
    parser.add_argument(
        "--output-microstructure",
        type=Path,
        default=Path("data/market/silver/microstructure/depth_features.csv"),
        help="CSV destination for microstructure features",
    )
    parser.add_argument(
        "--output-volatility",
        type=Path,
        default=Path("data/market/silver/volatility/risk_metrics.csv"),
        help="CSV destination for volatility features",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=25,
        help="Require at least N candles before emitting features",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_candles(input_path: Path) -> pd.DataFrame:
    """Load OHLC candle data from NDJSON file."""
    log(f"Loading candles from {input_path}")

    candles = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                candles.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(candles)

    if df.empty:
        log("ERROR: No candles loaded")
        return df

    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)

    log(f"Loaded {len(df)} candles")
    log(f"Date range: {df['time'].min()} to {df['time'].max()}")

    return df


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from OHLC candle data."""
    if len(df) < 2:
        return pd.DataFrame()

    result = df.copy()

    # Use close price as primary price, or compute mid from bid/ask
    if 'bid_close' in df.columns and 'ask_close' in df.columns:
        result['mid'] = (df['bid_close'] + df['ask_close']) / 2
        result['best_bid'] = df['bid_close']
        result['best_ask'] = df['ask_close']
    else:
        result['mid'] = df['close']
        result['best_bid'] = df['close']  # Approximate
        result['best_ask'] = df['close']  # Approximate

    # Returns at different horizons
    result['ret_1'] = result['mid'].pct_change(1)
    result['ret_5'] = result['mid'].pct_change(5)
    result['ret_10'] = result['mid'].pct_change(10)

    # Rolling volatility (20-period)
    result['roll_vol_20'] = result['ret_1'].rolling(20, min_periods=5).std()
    result['roll_vol_50'] = result['ret_1'].rolling(50, min_periods=10).std()

    # Rolling mean and z-score
    result['roll_mean_20'] = result['mid'].rolling(20, min_periods=5).mean()
    result['zscore_20'] = (result['mid'] - result['roll_mean_20']) / result['roll_vol_20']

    # EWMA (exponentially weighted moving average)
    result['ewma_short'] = result['mid'].ewm(span=5).mean()
    result['ewma_long'] = result['mid'].ewm(span=20).mean()
    result['ewma_signal'] = result['ewma_short'] - result['ewma_long']

    # Spread statistics (if available)
    if 'spread' in df.columns:
        result['spread_pct'] = df['spread'] / result['mid']
        result['spread_zscore'] = (df['spread'] - df['spread'].rolling(20).mean()) / df['spread'].rolling(20).std()

    # High-Low range
    if 'high' in df.columns and 'low' in df.columns:
        result['hl_range'] = df['high'] - df['low']
        result['hl_range_pct'] = result['hl_range'] / result['mid']

    # Volatility regimes
    vol_median = result['roll_vol_20'].median()
    result['high_vol_regime'] = result['roll_vol_20'] > (vol_median * 1.5)

    # Price momentum
    result['momentum_5'] = result['mid'] - result['mid'].shift(5)
    result['momentum_20'] = result['mid'] - result['mid'].shift(20)

    return result


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market microstructure features from candle data."""
    if len(df) < 2:
        return pd.DataFrame()

    result = df.copy()

    # Volume analysis
    if 'volume' in df.columns:
        result['volume_ma_20'] = df['volume'].rolling(20, min_periods=5).mean()
        result['volume_ratio'] = df['volume'] / result['volume_ma_20']
        result['volume_zscore'] = (df['volume'] - result['volume_ma_20']) / df['volume'].rolling(20, min_periods=5).std()

    # Spread analysis (if available)
    if 'spread' in df.columns and 'mid' in result.columns:
        result['effective_spread'] = df['spread'] / result['mid']
        result['spread_ma'] = df['spread'].rolling(20, min_periods=5).mean()

    # Bid-Ask imbalance (if available)
    if 'bid_close' in df.columns and 'ask_close' in df.columns:
        result['ba_imbalance'] = (df['ask_close'] - df['bid_close']) / result['mid']

    return result


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility and risk metrics from candle data."""
    if len(df) < 2:
        return pd.DataFrame()

    result = df.copy()

    # Close-to-close volatility
    if 'close' in df.columns:
        result['cc_vol_20'] = df['close'].pct_change().rolling(20, min_periods=5).std()
        result['cc_vol_50'] = df['close'].pct_change().rolling(50, min_periods=10).std()

    # Parkinson volatility (high-low range)
    if 'high' in df.columns and 'low' in df.columns:
        result['parkinson_vol'] = np.sqrt(
            1/(4 * np.log(2)) *
            np.log(df['high'] / df['low'])**2
        ).rolling(20, min_periods=5).mean()

    # Garman-Klass volatility (OHLC)
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        hl = np.log(df['high'] / df['low'])**2
        co = np.log(df['close'] / df['open'])**2
        result['gk_vol'] = np.sqrt(0.5 * hl - (2*np.log(2)-1) * co).rolling(20, min_periods=5).mean()

    # Realized volatility (intraday range)
    if 'high' in df.columns and 'low' in df.columns and 'mid' in result.columns:
        result['realized_vol'] = (df['high'] - df['low']) / result['mid']
        result['realized_vol_ma'] = result['realized_vol'].rolling(20, min_periods=5).mean()

    # Volatility of volatility
    if 'roll_vol_20' in result.columns:
        result['vol_of_vol'] = result['roll_vol_20'].rolling(20, min_periods=5).std()

    return result


def save_features(df: pd.DataFrame, output_path: Path, feature_type: str) -> int:
    """Save features to CSV file."""
    if df.empty:
        log(f"WARNING: No {feature_type} features to save")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop rows with too many NaNs (from rolling window initialization)
    min_required_cols = len(df.columns) * 0.5  # At least 50% non-NaN
    df_clean = df.dropna(thresh=min_required_cols)

    df_clean.to_csv(output_path, index=False)
    log(f"Saved {len(df_clean)} rows of {feature_type} features to {output_path}")

    return len(df_clean)


def log(message: str) -> None:
    """Emit structured progress messages to stderr."""
    sys.stderr.write(f"[build_market_features_from_candles] {message}\n")
    sys.stderr.flush()


def main(argv: Iterable[str] = None) -> None:
    """Main processing loop for historical candle feature engineering."""
    args = parse_args(argv)

    log("Starting historical market feature engineering")

    # Load candles
    df = load_candles(args.input)

    if df.empty or len(df) < args.min_rows:
        log(f"ERROR: Need at least {args.min_rows} candles, got {len(df)}")
        sys.exit(1)

    # Compute features (technical first as others depend on it)
    log("Computing technical features...")
    technical_df = compute_technical_features(df)

    log("Computing microstructure features...")
    microstructure_df = compute_microstructure_features(technical_df)

    log("Computing volatility features...")
    volatility_df = compute_volatility_features(technical_df)

    # Save features
    counts = {
        'technical': save_features(technical_df, args.output_technical, 'technical'),
        'microstructure': save_features(microstructure_df, args.output_microstructure, 'microstructure'),
        'volatility': save_features(volatility_df, args.output_volatility, 'volatility')
    }

    log(f"Finished: {len(df)} candles processed, {counts} rows written")


if __name__ == "__main__":
    main()
