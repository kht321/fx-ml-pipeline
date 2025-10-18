#!/usr/bin/env python3
"""Consolidate Silver-layer S&P 500 features into Gold-layer training data.

This is a simplified version of build_market_gold.py adapted for S&P 500 data
which doesn't have bid-ask spreads like forex markets.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def log(message: str):
    """Simple logging."""
    print(f"[build_sp500_gold] {message}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technical-features", type=Path, required=True)
    parser.add_argument("--microstructure-features", type=Path, required=True)
    parser.add_argument("--volatility-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-selection", choices=["all", "core", "extended"], default="all")
    return parser.parse_args()


def load_silver_features(tech_path: Path, micro_path: Path, vol_path: Path) -> tuple:
    """Load all Silver-layer features."""
    log("Loading Silver layer features")

    tech_df = pd.read_csv(tech_path)
    micro_df = pd.read_csv(micro_path)
    vol_df = pd.read_csv(vol_path)

    log(f"Loaded: {len(tech_df)} technical, {len(micro_df)} microstructure, {len(vol_df)} volatility observations")

    return tech_df, micro_df, vol_df


def merge_features(tech_df: pd.DataFrame, micro_df: pd.DataFrame, vol_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all Silver features on time."""
    log("Merging features")

    # Use time as the merge key
    merged = tech_df.copy()

    # Merge microstructure (skip overlapping columns except time)
    micro_cols = [c for c in micro_df.columns if c not in tech_df.columns or c == 'time']
    merged = merged.merge(micro_df[micro_cols], on='time', how='inner', suffixes=('', '_micro'))

    # Merge volatility (skip overlapping columns except time)
    vol_cols = [c for c in vol_df.columns if c not in merged.columns or c == 'time']
    merged = merged.merge(vol_df[vol_cols], on='time', how='inner', suffixes=('', '_vol'))

    log(f"Merged result: {len(merged)} observations, {len(merged.columns)} features")

    return merged


def select_features(df: pd.DataFrame, selection: str) -> pd.DataFrame:
    """Select features based on strategy."""
    log(f"Applying '{selection}' feature selection")

    if selection == "all":
        # Return all columns
        return df.copy()

    elif selection == "core":
        # Select core features for training
        core_features = [
            'time', 'instrument', 'mid', 'close',
            'ret_1', 'ret_5', 'ret_10',
            'roll_vol_20', 'roll_vol_50',
            'ewma_short', 'ewma_long', 'ewma_signal',
            'volume', 'volume_ma_20', 'volume_ratio',
            'hl_range', 'hl_range_pct',
            'momentum_5', 'momentum_20',
            'high_vol_regime',
        ]
        available = [c for c in core_features if c in df.columns]
        log(f"Core selection: {len(available)} features")
        return df[available].copy()

    elif selection == "extended":
        # Select extended feature set
        # Keep all numeric features plus key identifiers
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        meta_cols = ['time', 'instrument', 'granularity']
        keep_cols = list(set(numeric_cols + meta_cols))
        keep_cols = [c for c in df.columns if c in keep_cols]  # Preserve order
        log(f"Extended selection: {len(keep_cols)} features")
        return df[keep_cols].copy()

    return df


def main():
    """Main execution."""
    args = parse_args()

    # Load Silver features
    tech_df, micro_df, vol_df = load_silver_features(
        args.technical_features,
        args.microstructure_features,
        args.volatility_features
    )

    # Merge features
    merged_df = merge_features(tech_df, micro_df, vol_df)

    # Select features
    final_df = select_features(merged_df, args.feature_selection)

    # Remove rows with too many NaN values
    threshold = len(final_df.columns) * 0.5  # Keep rows with >50% non-NaN
    initial_rows = len(final_df)
    final_df = final_df.dropna(thresh=threshold)
    dropped = initial_rows - len(final_df)

    if dropped > 0:
        log(f"Dropped {dropped} rows with excessive missing values")

    # Save Gold features
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)

    # Also save Parquet for Feast (with event_timestamp)
    try:
        parquet_path = args.output.with_suffix('.parquet')
        df_parquet = final_df.copy()

        # Ensure event_timestamp column exists for Feast
        if 'event_timestamp' not in df_parquet.columns:
            if 'time' in df_parquet.columns:
                df_parquet['event_timestamp'] = pd.to_datetime(df_parquet['time'])

        # Ensure instrument column exists
        if 'instrument' not in df_parquet.columns:
            df_parquet['instrument'] = 'SPX500_USD'

        df_parquet.to_parquet(parquet_path, index=False)
        log(f"✓ Parquet saved for Feast: {parquet_path}")
    except Exception as e:
        log(f"⚠ Could not save Parquet (install pyarrow): {e}")

    log(f"✓ Gold layer saved: {args.output}")
    log(f"  Observations: {len(final_df):,}")
    log(f"  Features: {len(final_df.columns):,}")
    log(f"  Missing values: {final_df.isna().sum().sum():,} ({final_df.isna().sum().sum() / (len(final_df) * len(final_df.columns)) * 100:.2f}%)")


if __name__ == "__main__":
    main()
