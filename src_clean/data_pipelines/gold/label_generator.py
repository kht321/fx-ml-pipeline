"""
Label Generator for Price Prediction

Repository Location: fx-ml-pipeline/src_clean/data_pipelines/gold/label_generator.py

Purpose:
    Generates prediction labels for supervised learning.
    Creates labels for N-minute ahead price prediction (classification and regression).

Input:
    - Gold market features: data_clean/gold/market/features/*.csv

Output:
    - Gold labels: data_clean/gold/market/labels/*_labels_{horizon}min.csv

Label Types:
    - Classification: 1 if price(t+N) > price(t), else 0
    - Regression: price(t+N) - price(t)

Usage:
    python src_clean/data_pipelines/gold/label_generator.py \\
        --input data_clean/gold/market/features/spx500_features.csv \\
        --output data_clean/gold/market/labels/spx500_labels_30min.csv \\
        --horizon 30
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


class LabelGenerator:
    """Generates prediction labels for supervised learning."""

    def __init__(self, input_path: Path, output_path: Path, horizon_minutes: int = 30):
        """
        Initialize label generator.

        Parameters
        ----------
        input_path : Path
            Gold market features CSV
        horizon_minutes : int
            Prediction horizon in minutes
        output_path : Path
            Output path for labels
        """
        self.input_path = input_path
        self.output_path = output_path
        self.horizon_minutes = horizon_minutes

    def load_features(self) -> pd.DataFrame:
        """Load gold market features."""
        logger.info(f"Loading features from {self.input_path}")

        df = pd.read_csv(self.input_path)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values('time').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} samples")
        return df

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate prediction labels.

        Creates both classification and regression targets.
        """
        logger.info(f"Generating labels for {self.horizon_minutes}-minute prediction horizon...")

        result = df[['time', 'instrument', 'close']].copy()

        # Calculate future price (N minutes ahead)
        # For 1-minute candles, shift by N rows
        result['future_close'] = df['close'].shift(-self.horizon_minutes)

        # Classification label: 1 if price goes up, 0 if down
        result['target_classification'] = (result['future_close'] > df['close']).astype(int)

        # Regression label: Actual price change
        result['target_regression'] = result['future_close'] - df['close']

        # Price change percentage
        result['target_pct_change'] = (result['target_regression'] / df['close']) * 100

        # Additional labels for multi-class classification (optional)
        result['target_multiclass'] = pd.cut(
            result['target_pct_change'],
            bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
            labels=['strong_down', 'down', 'neutral', 'up', 'strong_up']
        )

        # Remove rows without future prices (end of dataset)
        valid_mask = result['future_close'].notna()
        result = result[valid_mask].copy()

        logger.info(f"Generated labels for {len(result)} samples")

        # Label distribution
        value_counts = result['target_classification'].value_counts()
        logger.info(f"Label distribution:")
        logger.info(f"  Up (1): {value_counts.get(1, 0)} ({value_counts.get(1, 0) / len(result) * 100:.1f}%)")
        logger.info(f"  Down (0): {value_counts.get(0, 0)} ({value_counts.get(0, 0) / len(result) * 100:.1f}%)")

        return result

    def create_stratified_folds(self, df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
        """
        Create time series folds for cross-validation.

        Uses expanding window approach to respect temporal order.
        """
        logger.info(f"Creating {n_folds} time series folds...")

        df = df.sort_values('time').reset_index(drop=True)

        # Calculate fold boundaries
        total_samples = len(df)
        fold_size = total_samples // (n_folds + 1)  # Reserve first fold for minimum training

        df['fold'] = -1

        for fold in range(n_folds):
            # Train on all data up to this point
            # Test on the next fold_size samples
            test_start = fold_size * (fold + 1)
            test_end = test_start + fold_size

            if test_end > total_samples:
                test_end = total_samples

            df.loc[test_start:test_end-1, 'fold'] = fold

        logger.info(f"Created {n_folds} folds with expanding window strategy")

        return df

    def save_labels(self, df: pd.DataFrame):
        """Save labels to gold layer."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        df['prediction_horizon_minutes'] = self.horizon_minutes
        df['label_generated_at'] = pd.Timestamp.now(tz='UTC').isoformat()

        # Save
        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved {len(df)} labels to {self.output_path}")

        # Save Parquet for Feast
        try:
            parquet_path = self.output_path.with_suffix('.parquet')
            df_parquet = df.copy()

            if 'event_timestamp' not in df_parquet.columns:
                df_parquet['event_timestamp'] = df_parquet['time']

            df_parquet.to_parquet(parquet_path, index=False)
            logger.info(f"Saved Parquet: {parquet_path}")
        except Exception as e:
            logger.warning(f"Could not save Parquet: {e}")

    def run(self, create_folds: bool = True):
        """Execute label generation."""
        logger.info("="*80)
        logger.info(f"Label Generator - {self.horizon_minutes}-Minute Prediction")
        logger.info("="*80)

        # Load features
        df = self.load_features()

        # Generate labels
        labels_df = self.generate_labels(df)

        # Create CV folds if requested
        if create_folds:
            labels_df = self.create_stratified_folds(labels_df)

        # Save
        self.save_labels(labels_df)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("LABEL GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Prediction horizon: {self.horizon_minutes} minutes")
        logger.info(f"Total samples: {len(labels_df)}")
        logger.info(f"Classification labels: target_classification (0/1)")
        logger.info(f"Regression labels: target_regression (price change)")
        logger.info(f"Multi-class labels: target_multiclass (5 classes)")
        logger.info(f"Output: {self.output_path}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Gold features CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output labels CSV")
    parser.add_argument("--horizon", type=int, default=30, help="Prediction horizon in minutes")
    parser.add_argument("--no-folds", action="store_true", help="Skip CV fold creation")
    args = parser.parse_args()

    generator = LabelGenerator(
        input_path=args.input,
        output_path=args.output,
        horizon_minutes=args.horizon
    )

    generator.run(create_folds=not args.no_folds)


if __name__ == "__main__":
    main()
