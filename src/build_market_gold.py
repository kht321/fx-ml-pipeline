"""Transform market Silver features into Gold-layer training datasets.

This script consolidates the various Silver-layer market features (technical,
microstructure, volatility) into unified Gold-layer training datasets optimized
for ML model consumption.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Build the argument parser for market Gold layer processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--technical-features",
        type=Path,
        default=Path("data/market/silver/technical_features/sgd_vs_majors.csv"),
        help="CSV containing technical analysis features",
    )
    parser.add_argument(
        "--microstructure-features",
        type=Path,
        default=Path("data/market/silver/microstructure/depth_features.csv"),
        help="CSV containing market microstructure features",
    )
    parser.add_argument(
        "--volatility-features",
        type=Path,
        default=Path("data/market/silver/volatility/risk_metrics.csv"),
        help="CSV containing volatility and risk features",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/market/gold/training/market_features.csv"),
        help="Destination CSV for Gold market training data",
    )
    parser.add_argument(
        "--min-obs-per-instrument",
        type=int,
        default=100,
        help="Minimum observations required per instrument",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["all", "core", "minimal"],
        default="all",
        help="Feature selection strategy",
    )
    return parser.parse_args(list(argv))


def load_silver_features(technical_path: Path,
                        microstructure_path: Path,
                        volatility_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all Silver layer feature files."""

    dfs = {}

    # Load technical features
    if technical_path.exists():
        technical_df = pd.read_csv(technical_path)
        technical_df['time'] = pd.to_datetime(technical_df['time'])
        dfs['technical'] = technical_df
    else:
        dfs['technical'] = pd.DataFrame()

    # Load microstructure features
    if microstructure_path.exists():
        microstructure_df = pd.read_csv(microstructure_path)
        microstructure_df['time'] = pd.to_datetime(microstructure_df['time'])
        dfs['microstructure'] = microstructure_df
    else:
        dfs['microstructure'] = pd.DataFrame()

    # Load volatility features
    if volatility_path.exists():
        volatility_df = pd.read_csv(volatility_path)
        volatility_df['time'] = pd.to_datetime(volatility_df['time'])
        dfs['volatility'] = volatility_df
    else:
        dfs['volatility'] = pd.DataFrame()

    return dfs['technical'], dfs['microstructure'], dfs['volatility']


def merge_market_features(technical_df: pd.DataFrame,
                         microstructure_df: pd.DataFrame,
                         volatility_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all market features on time and instrument."""

    # Start with technical features as base
    if technical_df.empty:
        return pd.DataFrame()

    merged_df = technical_df.copy()

    # Merge microstructure features
    if not microstructure_df.empty:
        micro_cols = [col for col in microstructure_df.columns
                     if col not in ['time', 'instrument', 'y']]
        micro_merge = microstructure_df[['time', 'instrument'] + micro_cols]

        merged_df = merged_df.merge(
            micro_merge,
            on=['time', 'instrument'],
            how='left',
            suffixes=('', '_micro')
        )

    # Merge volatility features
    if not volatility_df.empty:
        vol_cols = [col for col in volatility_df.columns
                   if col not in ['time', 'instrument', 'y']]
        vol_merge = volatility_df[['time', 'instrument'] + vol_cols]

        merged_df = merged_df.merge(
            vol_merge,
            on=['time', 'instrument'],
            how='left',
            suffixes=('', '_vol')
        )

    return merged_df


def select_features(df: pd.DataFrame, strategy: str = "all") -> pd.DataFrame:
    """Select features based on strategy."""

    if df.empty:
        return df

    # Core columns to always keep
    core_cols = ['time', 'instrument', 'mid', 'spread', 'ret_1', 'ret_5', 'y']

    if strategy == "minimal":
        # Only keep core price features
        feature_cols = [
            'roll_vol_20', 'zscore_20', 'bid_liquidity', 'ask_liquidity'
        ]
    elif strategy == "core":
        # Technical + basic microstructure
        feature_cols = [
            'roll_vol_20', 'zscore_20', 'ewma_signal', 'spread_pct',
            'bid_liquidity', 'ask_liquidity', 'total_liquidity',
            'effective_spread', 'vol_20', 'high_vol_regime'
        ]
    else:  # "all"
        # All available features
        feature_cols = [col for col in df.columns if col not in core_cols]

    # Keep only existing columns
    available_cols = core_cols + [col for col in feature_cols if col in df.columns]

    return df[available_cols].copy()


def clean_and_validate(df: pd.DataFrame, min_obs_per_instrument: int = 100) -> pd.DataFrame:
    """Clean data and apply validation rules."""

    if df.empty:
        return df

    # Remove instruments with insufficient data
    instrument_counts = df['instrument'].value_counts()
    valid_instruments = instrument_counts[instrument_counts >= min_obs_per_instrument].index
    df = df[df['instrument'].isin(valid_instruments)].copy()

    if df.empty:
        return df

    # Remove rows with missing target
    df = df.dropna(subset=['y']).copy()

    # Handle infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Fill missing values with forward fill then median
    for col in numeric_cols:
        if col not in ['time', 'y']:
            df[col] = df.groupby('instrument')[col].fillna(method='ffill')
            df[col] = df[col].fillna(df[col].median())

    # Remove any remaining rows with NaN in features
    feature_cols = [col for col in df.columns if col not in ['time', 'instrument', 'y']]
    df = df.dropna(subset=feature_cols).copy()

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Gold-layer derived features."""

    if df.empty:
        return df

    df = df.copy()

    # Cross-instrument features (if multiple instruments)
    instruments = df['instrument'].unique()

    if len(instruments) > 1:
        # Create time-based pivot for cross-correlations
        pivot_returns = df.pivot_table(
            index='time',
            columns='instrument',
            values='ret_1',
            fill_value=0
        )

        # Add correlation features for USD_SGD specifically
        if 'USD_SGD' in pivot_returns.columns:
            # Correlation with EUR_USD
            if 'EUR_USD' in pivot_returns.columns:
                corr_window = 50
                rolling_corr = pivot_returns['USD_SGD'].rolling(corr_window).corr(
                    pivot_returns['EUR_USD']
                )

                # Merge back to main dataframe
                corr_df = pd.DataFrame({
                    'time': rolling_corr.index,
                    'usd_sgd_eur_usd_corr': rolling_corr.values
                })

                df = df.merge(corr_df, on='time', how='left')

            # Relative performance vs basket
            basket_return = pivot_returns.mean(axis=1)
            rel_perf = pd.DataFrame({
                'time': pivot_returns.index,
                'relative_performance': pivot_returns['USD_SGD'] - basket_return
            })

            df = df.merge(rel_perf, on='time', how='left')

    # Time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Market session indicators (approximate)
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

    return df


def log(message: str) -> None:
    """Emit structured progress messages to stderr."""
    sys.stderr.write(f"[build_market_gold] {message}\n")
    sys.stderr.flush()


def main(argv: Iterable[str] | None = None) -> None:
    """Main processing function for market Gold layer."""
    args = parse_args(argv or sys.argv[1:])

    log("Loading Silver layer market features")

    # Load all Silver features
    technical_df, microstructure_df, volatility_df = load_silver_features(
        args.technical_features,
        args.microstructure_features,
        args.volatility_features
    )

    if technical_df.empty:
        log("No technical features found - cannot proceed")
        sys.exit(1)

    log(f"Loaded: {len(technical_df)} technical, {len(microstructure_df)} microstructure, "
        f"{len(volatility_df)} volatility observations")

    # Merge all features
    log("Merging market features")
    merged_df = merge_market_features(technical_df, microstructure_df, volatility_df)

    if merged_df.empty:
        log("No data after merging - cannot proceed")
        sys.exit(1)

    # Feature selection
    log(f"Applying {args.feature_selection} feature selection")
    selected_df = select_features(merged_df, args.feature_selection)

    # Add derived features
    log("Adding Gold layer derived features")
    enhanced_df = add_derived_features(selected_df)

    # Clean and validate
    log("Cleaning and validating data")
    final_df = clean_and_validate(enhanced_df, args.min_obs_per_instrument)

    if final_df.empty:
        log("No data remaining after cleaning - check data quality")
        sys.exit(1)

    # Sort by time and instrument
    final_df = final_df.sort_values(['instrument', 'time']).reset_index(drop=True)

    # Save to Gold layer
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)

    # Summary statistics
    instruments = final_df['instrument'].unique()
    feature_count = len([col for col in final_df.columns if col not in ['time', 'instrument', 'y']])

    log(f"Gold layer complete: {len(final_df)} observations, {len(instruments)} instruments, "
        f"{feature_count} features")

    for instrument in instruments:
        inst_count = len(final_df[final_df['instrument'] == instrument])
        target_mean = final_df[final_df['instrument'] == instrument]['y'].mean()
        log(f"  {instrument}: {inst_count} obs, target rate: {target_mean:.3f}")


if __name__ == "__main__":
    main()