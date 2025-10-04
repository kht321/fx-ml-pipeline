"""Transform market Bronze tick data into Silver-level technical features.

This module is part of the Market Data medallion pipeline. It reads streaming
tick JSON from the Bronze layer, computes per-instrument technical features,
and writes the results incrementally to the Silver layer for market microstructure analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import numpy as np


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Define the command-line interface and parse user inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to newline-delimited tick JSON (defaults to stdin)",
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
        "--horizon",
        type=int,
        default=5,
        help="Tick horizon for the binary target label",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=25,
        help="Require at least N price rows before emitting features",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=100,
        help="Flush engineered rows every N price ticks",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log progress every N price ticks (0 to disable)",
    )
    return parser.parse_args(list(argv))


def parse_price_tick(line: str) -> dict | None:
    """Parse a single tick JSON line and extract price data."""
    try:
        tick = json.loads(line.strip())
        if tick.get("type") != "PRICE":
            return None

        # Extract bid/ask data
        bids = tick.get("bids", [])
        asks = tick.get("asks", [])

        if not bids or not asks:
            return None

        # Top of book
        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        bid_liquidity = float(bids[0]["liquidity"])
        ask_liquidity = float(asks[0]["liquidity"])

        return {
            "time": pd.to_datetime(tick["time"]),
            "instrument": tick["instrument"],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": (best_bid + best_ask) / 2,
            "spread": best_ask - best_bid,
            "bid_liquidity": bid_liquidity,
            "ask_liquidity": ask_liquidity,
            "tradeable": tick.get("tradeable", False),
            "status": tick.get("status", "unknown")
        }

    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
        return None


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from price data."""
    if len(df) < 2:
        return df

    # Sort by time to ensure proper ordering
    df = df.sort_values('time').copy()

    # Returns
    df['ret_1'] = df['mid'].pct_change(1)
    df['ret_5'] = df['mid'].pct_change(5)

    # Rolling statistics (20-period)
    df['roll_vol_20'] = df['ret_1'].rolling(20, min_periods=5).std()
    df['roll_mean_20'] = df['mid'].rolling(20, min_periods=5).mean()
    df['zscore_20'] = (df['mid'] - df['roll_mean_20']) / df['roll_vol_20']

    # EWMA (exponentially weighted moving average)
    df['ewma_short'] = df['mid'].ewm(span=5).mean()
    df['ewma_long'] = df['mid'].ewm(span=20).mean()
    df['ewma_signal'] = df['ewma_short'] - df['ewma_long']

    # Spread statistics
    df['spread_pct'] = df['spread'] / df['mid']
    df['spread_zscore'] = (df['spread'] - df['spread'].rolling(20).mean()) / df['spread'].rolling(20).std()

    return df


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market microstructure features."""
    if len(df) < 2:
        return df

    df = df.sort_values('time').copy()

    # Liquidity metrics
    df['total_liquidity'] = df['bid_liquidity'] + df['ask_liquidity']
    df['liquidity_imbalance'] = (df['ask_liquidity'] - df['bid_liquidity']) / df['total_liquidity']

    # Bid-ask spread analysis
    df['effective_spread'] = df['spread'] / df['mid']
    df['quoted_depth'] = np.minimum(df['bid_liquidity'], df['ask_liquidity'])

    # Rolling liquidity statistics
    df['avg_liquidity_20'] = df['total_liquidity'].rolling(20, min_periods=5).mean()
    df['liquidity_shock'] = (df['total_liquidity'] - df['avg_liquidity_20']) / df['avg_liquidity_20']

    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility and risk metrics."""
    if len(df) < 2:
        return df

    df = df.sort_values('time').copy()

    # Realized volatility (different windows)
    df['vol_5'] = df['ret_1'].rolling(5, min_periods=2).std() * np.sqrt(5)
    df['vol_20'] = df['ret_1'].rolling(20, min_periods=5).std() * np.sqrt(20)
    df['vol_60'] = df['ret_1'].rolling(60, min_periods=10).std() * np.sqrt(60)

    # Range-based volatility
    df['high_5'] = df['mid'].rolling(5, min_periods=2).max()
    df['low_5'] = df['mid'].rolling(5, min_periods=2).min()
    df['range_vol'] = (df['high_5'] - df['low_5']) / df['mid']

    # Volatility regime indicators
    df['vol_percentile'] = df['vol_20'].rolling(100, min_periods=20).rank(pct=True)
    df['high_vol_regime'] = (df['vol_percentile'] > 0.8).astype(int)
    df['low_vol_regime'] = (df['vol_percentile'] < 0.2).astype(int)

    return df


def create_target_label(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create binary target label for price direction prediction."""
    if len(df) < horizon + 1:
        df['y'] = np.nan
        return df

    df = df.sort_values('time').copy()

    # Future price after horizon ticks
    df['future_mid'] = df['mid'].shift(-horizon)
    df['y'] = (df['future_mid'] > df['mid']).astype(int)

    # Remove rows where we can't compute the target
    df.loc[df.index[-horizon:], 'y'] = np.nan

    return df


def process_ticks_to_features(ticks: List[dict], horizon: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process a batch of ticks into all feature types."""
    if not ticks:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Create base dataframe
    df = pd.DataFrame(ticks)

    # Process each instrument separately
    technical_dfs = []
    microstructure_dfs = []
    volatility_dfs = []

    for instrument in df['instrument'].unique():
        instrument_df = df[df['instrument'] == instrument].copy()

        if len(instrument_df) < 2:
            continue

        # Compute different feature types
        tech_df = compute_technical_features(instrument_df)
        micro_df = compute_microstructure_features(instrument_df)
        vol_df = compute_volatility_features(instrument_df)

        # Add target label
        tech_df = create_target_label(tech_df, horizon)
        micro_df = create_target_label(micro_df, horizon)
        vol_df = create_target_label(vol_df, horizon)

        technical_dfs.append(tech_df)
        microstructure_dfs.append(micro_df)
        volatility_dfs.append(vol_df)

    # Combine all instruments
    technical_features = pd.concat(technical_dfs, ignore_index=True) if technical_dfs else pd.DataFrame()
    microstructure_features = pd.concat(microstructure_dfs, ignore_index=True) if microstructure_dfs else pd.DataFrame()
    volatility_features = pd.concat(volatility_dfs, ignore_index=True) if volatility_dfs else pd.DataFrame()

    return technical_features, microstructure_features, volatility_features


def append_to_csv(df: pd.DataFrame, output_path: Path, exclude_cols: set = None):
    """Append new rows to CSV file, handling file creation and headers."""
    if df.empty:
        return 0

    exclude_cols = exclude_cols or set()
    df_to_write = df[[col for col in df.columns if col not in exclude_cols]].copy()

    # Drop rows with NaN in target
    if 'y' in df_to_write.columns:
        df_to_write = df_to_write.dropna(subset=['y'])

    if df_to_write.empty:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with header if file doesn't exist
    write_header = not output_path.exists()
    df_to_write.to_csv(output_path, mode='a', header=write_header, index=False)

    return len(df_to_write)


def log(message: str) -> None:
    """Emit structured progress messages to stderr."""
    sys.stderr.write(f"[build_market_features] {message}\n")
    sys.stderr.flush()


def main(argv: Iterable[str] | None = None) -> None:
    """Main processing loop for market feature engineering."""
    args = parse_args(argv or sys.argv[1:])

    input_stream = args.input.open('r') if args.input else sys.stdin

    tick_buffer = []
    total_ticks = 0
    total_written = {'technical': 0, 'microstructure': 0, 'volatility': 0}

    log(f"Starting market feature engineering (flush every {args.flush_interval} ticks)")

    try:
        for line_num, line in enumerate(input_stream, 1):
            line = line.strip()
            if not line:
                continue

            # Parse tick
            tick_data = parse_price_tick(line)
            if tick_data is None:
                continue

            tick_buffer.append(tick_data)
            total_ticks += 1

            # Process buffer when it reaches flush interval
            if len(tick_buffer) >= args.flush_interval:
                # Skip if we don't have minimum rows
                if len(tick_buffer) >= args.min_rows:
                    technical, microstructure, volatility = process_ticks_to_features(
                        tick_buffer, args.horizon
                    )

                    # Write to separate CSV files
                    tech_written = append_to_csv(technical, args.output_technical)
                    micro_written = append_to_csv(microstructure, args.output_microstructure)
                    vol_written = append_to_csv(volatility, args.output_volatility)

                    total_written['technical'] += tech_written
                    total_written['microstructure'] += micro_written
                    total_written['volatility'] += vol_written

                    if args.log_every and total_ticks % args.log_every == 0:
                        log(f"processed {total_ticks} ticks, written {total_written}")

                # Clear buffer but keep some overlap for rolling calculations
                overlap_size = max(60, args.min_rows)  # Keep 60 ticks for rolling features
                tick_buffer = tick_buffer[-overlap_size:] if len(tick_buffer) > overlap_size else []

    except KeyboardInterrupt:
        log("interrupted by user")
    finally:
        # Process remaining ticks
        if len(tick_buffer) >= args.min_rows:
            technical, microstructure, volatility = process_ticks_to_features(
                tick_buffer, args.horizon
            )

            tech_written = append_to_csv(technical, args.output_technical)
            micro_written = append_to_csv(microstructure, args.output_microstructure)
            vol_written = append_to_csv(volatility, args.output_volatility)

            total_written['technical'] += tech_written
            total_written['microstructure'] += micro_written
            total_written['volatility'] += vol_written

        log(f"finished: {total_ticks} ticks processed, {total_written} rows written")

        if input_stream != sys.stdin:
            input_stream.close()


if __name__ == "__main__":
    main()