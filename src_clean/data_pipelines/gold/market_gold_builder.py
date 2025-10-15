"""
Market Gold Layer Builder

Repository Location: fx-ml-pipeline/src_clean/data_pipelines/gold/market_gold_builder.py

Purpose:
    Merges all silver layer market features (technical, microstructure, volatility)
    into a single training-ready gold layer dataset.

Input:
    - Silver technical: data_clean/silver/market/technical/*.csv
    - Silver microstructure: data_clean/silver/market/microstructure/*.csv
    - Silver volatility: data_clean/silver/market/volatility/*.csv

Output:
    - Gold market features: data_clean/gold/market/features/*.csv
    - Gold market features: data_clean/gold/market/features/*.parquet (for Feast)

Usage:
    python src_clean/data_pipelines/gold/market_gold_builder.py \\
        --technical data_clean/silver/market/technical/spx500_technical.csv \\
        --microstructure data_clean/silver/market/microstructure/spx500_microstructure.csv \\
        --volatility data_clean/silver/market/volatility/spx500_volatility.csv \\
        --output data_clean/gold/market/features/spx500_features.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketGoldBuilder:
    """Builds gold layer market features by merging all silver features."""

    def __init__(
        self,
        technical_path: Path,
        microstructure_path: Path,
        volatility_path: Path,
        output_path: Path
    ):
        self.technical_path = technical_path
        self.microstructure_path = microstructure_path
        self.volatility_path = volatility_path
        self.output_path = output_path

    def load_silver_features(self) -> tuple:
        """Load all silver layer features."""
        logger.info("Loading silver layer features...")

        tech_df = pd.read_csv(self.technical_path)
        tech_df['time'] = pd.to_datetime(tech_df['time'], utc=True)

        micro_df = pd.read_csv(self.microstructure_path)
        micro_df['time'] = pd.to_datetime(micro_df['time'], utc=True)

        vol_df = pd.read_csv(self.volatility_path)
        vol_df['time'] = pd.to_datetime(vol_df['time'], utc=True)

        logger.info(f"Loaded: {len(tech_df)} technical, {len(micro_df)} microstructure, {len(vol_df)} volatility rows")

        return tech_df, micro_df, vol_df

    def merge_features(self, tech_df: pd.DataFrame, micro_df: pd.DataFrame, vol_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all silver features on time."""
        logger.info("Merging features...")

        # Start with technical features
        merged = tech_df.copy()

        # Get non-overlapping columns from microstructure
        micro_cols = ['time'] + [c for c in micro_df.columns if c not in tech_df.columns or c == 'time']
        merged = merged.merge(micro_df[micro_cols], on='time', how='inner', suffixes=('', '_micro'))

        # Get non-overlapping columns from volatility
        vol_cols = ['time'] + [c for c in vol_df.columns if c not in merged.columns or c == 'time']
        merged = merged.merge(vol_df[vol_cols], on='time', how='inner', suffixes=('', '_vol'))

        logger.info(f"Merged result: {len(merged)} rows, {len(merged.columns)} columns")

        return merged

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and clean features for training."""
        logger.info("Selecting and cleaning features...")

        # Remove rows with too many NaN values
        threshold = len(df.columns) * 0.7  # Keep rows with >70% non-NaN
        initial_rows = len(df)
        df = df.dropna(thresh=threshold)
        dropped = initial_rows - len(df)

        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with excessive missing values")

        # Forward fill remaining NaN values
        df = df.fillna(method='ffill')

        # Drop any remaining NaN rows
        df = df.dropna()

        logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")

        return df

    def save_gold_features(self, df: pd.DataFrame):
        """Save gold features in both CSV and Parquet formats."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save CSV
        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved CSV: {self.output_path}")

        # Save Parquet for Feast
        try:
            parquet_path = self.output_path.with_suffix('.parquet')

            # Ensure event_timestamp column for Feast
            df_parquet = df.copy()
            if 'event_timestamp' not in df_parquet.columns:
                df_parquet['event_timestamp'] = df_parquet['time']

            df_parquet.to_parquet(parquet_path, index=False)
            logger.info(f"Saved Parquet: {parquet_path}")
        except Exception as e:
            logger.warning(f"Could not save Parquet (install pyarrow): {e}")

    def run(self):
        """Execute the gold layer build."""
        logger.info("="*80)
        logger.info("Market Gold Layer Builder")
        logger.info("="*80)

        # Load silver features
        tech_df, micro_df, vol_df = self.load_silver_features()

        # Merge
        merged_df = self.merge_features(tech_df, micro_df, vol_df)

        # Select and clean
        gold_df = self.select_features(merged_df)

        # Save
        self.save_gold_features(gold_df)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("GOLD LAYER BUILD COMPLETE")
        logger.info("="*80)
        logger.info(f"Features: {len(gold_df.columns)}")
        logger.info(f"Samples: {len(gold_df)}")
        logger.info(f"Date range: {gold_df['time'].min()} to {gold_df['time'].max()}")
        logger.info(f"Output: {self.output_path}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technical", type=Path, required=True)
    parser.add_argument("--microstructure", type=Path, required=True)
    parser.add_argument("--volatility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    builder = MarketGoldBuilder(
        args.technical,
        args.microstructure,
        args.volatility,
        args.output
    )

    builder.run()


if __name__ == "__main__":
    main()
