"""
XGBoost Training Pipeline - 30-Minute Price Prediction

Repository Location: fx-ml-pipeline/src_clean/training/xgboost_training_pipeline.py

Purpose:
    Trains XGBoost models to predict S&P 500 price movement in next 30 minutes.
    Uses Gold layer features and labels (no label recreation).

    ARCHITECTURAL IMPROVEMENTS (2025-10):
    - ✓ Uses pre-computed Gold labels (no label recreation)
    - ✓ Proper train/val/test/OOT splits (60/20/10/10)
    - ✓ Model trained only on train+val, evaluated on unseen test+OOT

Target Variable:
    - Binary classification: Price up/down in next 30 minutes
    - Regression: Percentage returns (stationary target)

Features:
    - Market: Technical indicators, microstructure, volatility
    - News: Sentiment signals, trading signals (optional)

Output:
    - Trained XGBoost model (pkl file)
    - Feature importance rankings
    - Evaluation metrics on Train/Val/Test/OOT splits
    - Training visualizations

Usage:
    # Auto-infer labels path
    python src_clean/training/xgboost_training_pipeline.py \\
        --market-features data_clean/gold/market/features/spx500_features.csv \\
        --prediction-horizon 30 \\
        --task classification

    # Explicit labels path
    python src_clean/training/xgboost_training_pipeline.py \\
        --market-features data_clean/gold/market/features/spx500_features.csv \\
        --labels data_clean/gold/market/labels/spx500_labels_30min.csv \\
        --task classification
"""

import argparse
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_recall_fscore_support, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostTrainingPipeline:
    """Training pipeline for XGBoost price prediction models."""

    def __init__(
        self,
        market_features_path: Path,
        labels_path: Optional[Path] = None,
        news_signals_path: Optional[Path] = None,
        prediction_horizon_minutes: int = 30,
        output_dir: Path = Path("data_clean/models"),
        task: str = "classification"
    ):
        """
        Initialize training pipeline.

        Parameters
        ----------
        market_features_path : Path
            Path to Gold layer market features CSV
        labels_path : Path, optional
            Path to Gold layer labels CSV (if not provided, will be inferred)
        news_signals_path : Path, optional
            Path to Gold layer news signals CSV
        prediction_horizon_minutes : int
            Number of minutes ahead to predict (default: 30)
        output_dir : Path
            Directory to save trained models and outputs
        task : str
            'classification' or 'regression'
        """
        self.market_features_path = market_features_path
        self.labels_path = labels_path
        self.news_signals_path = news_signals_path
        self.prediction_horizon = prediction_horizon_minutes
        self.output_dir = output_dir
        self.task = task

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model configuration
        self.xgb_params = {
            "classification": {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "tree_method": "hist"
            },
            "regression": {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "tree_method": "hist"
            }
        }

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load market features, labels, and news signals from Gold layer."""
        logger.info("Loading data from Gold layer...")

        # Load market features
        market_df = pd.read_csv(self.market_features_path)
        market_df['time'] = pd.to_datetime(market_df['time'])
        logger.info(f"Loaded market features: {len(market_df)} rows, {len(market_df.columns)} columns")

        # Load or infer labels path
        if self.labels_path is None:
            # Infer labels path from features path
            features_dir = self.market_features_path.parent
            base_dir = features_dir.parent.parent.parent  # Go up to data_clean
            instrument = self.market_features_path.stem.replace('_features', '')
            self.labels_path = base_dir / f"gold/market/labels/{instrument}_labels_{self.prediction_horizon}min.csv"
            logger.info(f"Inferred labels path: {self.labels_path}")

        # Load labels
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}\n"
                f"Please run label_generator.py first or provide --labels"
            )

        labels_df = pd.read_csv(self.labels_path)
        labels_df['time'] = pd.to_datetime(labels_df['time'])
        logger.info(f"Loaded gold labels: {len(labels_df)} rows, {len(labels_df.columns)} columns")

        # Verify label prediction horizon matches
        if 'prediction_horizon_minutes' in labels_df.columns:
            label_horizon = labels_df['prediction_horizon_minutes'].iloc[0]
            if label_horizon != self.prediction_horizon:
                logger.warning(
                    f"Label horizon ({label_horizon}min) != requested horizon ({self.prediction_horizon}min). "
                    f"Using labels with {label_horizon}min horizon."
                )
                self.prediction_horizon = int(label_horizon)

        # Merge market features with labels on time
        logger.info("Merging market features with gold labels...")
        merged_df = pd.merge(
            market_df,
            labels_df[['time', 'target_classification', 'target_regression', 'target_pct_change', 'fold']],
            on='time',
            how='inner'
        )
        logger.info(f"Merged dataset: {len(merged_df)} rows after merging")

        # Map task to appropriate target column
        if self.task == "classification":
            merged_df['target'] = merged_df['target_classification']
        else:
            # For regression, use percentage change (stationary)
            merged_df['target'] = merged_df['target_pct_change']

        # Load news signals if available
        news_df = None
        if self.news_signals_path and self.news_signals_path.exists():
            news_df = pd.read_csv(self.news_signals_path)
            news_df['signal_time'] = pd.to_datetime(news_df['signal_time'])
            logger.info(f"Loaded news signals: {len(news_df)} rows, {len(news_df.columns)} columns")
        else:
            logger.warning("No news signals provided - using market features only")

        return merged_df, news_df

    def validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that labels are present and log statistics."""
        logger.info(f"Validating labels for {self.prediction_horizon}-minute prediction...")

        if 'target' not in df.columns:
            raise ValueError("Target column not found in dataframe. Labels should be loaded from Gold layer.")

        # Remove any rows with missing targets
        df = df.dropna(subset=['target'])

        logger.info(f"Valid samples with labels: {len(df)}")

        if self.task == "classification":
            value_counts = df['target'].value_counts()
            logger.info(f"Label distribution: Up={value_counts.get(1, 0)}, Down={value_counts.get(0, 0)}")
        else:
            logger.info(f"Target stats: mean={df['target'].mean():.6f}, std={df['target'].std():.6f}")

        return df

    def merge_market_news(
        self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        tolerance_hours: int = 6
    ) -> pd.DataFrame:
        """
        Merge market features with news signals using as-of join.

        Parameters
        ----------
        market_df : pd.DataFrame
            Market features with 'time' column
        news_df : pd.DataFrame, optional
            News signals with 'signal_time' column
        tolerance_hours : int
            Maximum lookback window for news (hours)
        """
        if news_df is None or news_df.empty:
            logger.info("No news data - using market features only")
            return market_df

        logger.info("Merging market features with news signals...")

        merged_rows = []
        tolerance = pd.Timedelta(hours=tolerance_hours)

        # Sort both dataframes
        market_df = market_df.sort_values('time')
        news_df = news_df.sort_values('signal_time')

        # Rename news columns to avoid conflicts
        news_features = [
            'signal_time', 'avg_sentiment', 'signal_strength',
            'trading_signal', 'article_count', 'quality_score'
        ]
        available_news = [c for c in news_features if c in news_df.columns]

        for _, market_row in market_df.iterrows():
            market_time = market_row['time']

            # Find most recent news within tolerance
            news_cutoff = market_time - tolerance
            eligible_news = news_df[
                (news_df['signal_time'] <= market_time) &
                (news_df['signal_time'] >= news_cutoff)
            ]

            merged_row = market_row.to_dict()

            if not eligible_news.empty:
                latest_news = eligible_news.iloc[-1]

                # Add news features with 'news_' prefix
                for col in available_news:
                    if col != 'signal_time':
                        merged_row[f'news_{col}'] = latest_news[col]

                news_age_minutes = (market_time - latest_news['signal_time']).total_seconds() / 60
                merged_row['news_age_minutes'] = news_age_minutes
                merged_row['news_available'] = 1
            else:
                # No news available - use neutral defaults
                for col in available_news:
                    if col != 'signal_time':
                        merged_row[f'news_{col}'] = 0.0

                merged_row['news_age_minutes'] = np.nan
                merged_row['news_available'] = 0

            merged_rows.append(merged_row)

        combined_df = pd.DataFrame(merged_rows)

        logger.info(f"Merged dataset: {len(combined_df)} observations")
        news_coverage = combined_df['news_available'].mean()
        logger.info(f"News coverage: {news_coverage:.1%}")

        return combined_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, list]:
        """
        Prepare features for training, preserving time index for splits.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        time_index : pd.Series
            Time column for temporal splits
        feature_names : list
            List of feature names
        """
        logger.info("Preparing features...")

        # Exclude all label-related and metadata columns
        exclude_cols = {
            'time', 'instrument', 'granularity', 'close',
            'target', 'target_classification', 'target_regression',
            'target_pct_change', 'target_multiclass', 'future_close',
            'signal_time', 'collected_at', 'event_timestamp',
            'prediction_horizon_minutes', 'label_generated_at', 'fold'
        }

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy()
        y = df['target'].copy()
        time_index = df['time'].copy()  # Preserve time for temporal splits

        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes

        # Fill missing values
        X = X.fillna(X.median())

        # Remove constant columns
        constant_cols = X.columns[X.nunique() <= 1]
        if len(constant_cols) > 0:
            logger.info(f"Removing {len(constant_cols)} constant columns")
            X = X.drop(constant_cols, axis=1)

        logger.info(f"Final feature set: {len(X.columns)} features, {len(X)} samples")

        return X, y, time_index, list(X.columns)

    def split_train_val_test_oot(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_index: pd.Series,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        oot_ratio: float = 0.10
    ) -> Dict:
        """
        Create time-based splits for Train/Val/Test/OOT.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        time_index : pd.Series
            Time column for temporal ordering
        train_ratio : float
            Proportion for training (default: 0.60)
        val_ratio : float
            Proportion for validation (default: 0.20)
        test_ratio : float
            Proportion for test set (default: 0.10)
        oot_ratio : float
            Proportion for out-of-time holdout (default: 0.10)

        Returns
        -------
        dict
            Dictionary with train/val/test/oot indices and date ranges
        """
        logger.info("\n" + "="*80)
        logger.info("TEMPORAL DATA SPLITTING")
        logger.info("="*80)

        # Sort by time
        sorted_idx = time_index.sort_values().index
        X_sorted = X.loc[sorted_idx]
        y_sorted = y.loc[sorted_idx]
        time_sorted = time_index.loc[sorted_idx]

        n = len(X_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        test_end = int(n * (train_ratio + val_ratio + test_ratio))

        splits = {
            'train_idx': X_sorted.index[:train_end],
            'val_idx': X_sorted.index[train_end:val_end],
            'test_idx': X_sorted.index[val_end:test_end],
            'oot_idx': X_sorted.index[test_end:],
            'train_dates': (time_sorted.iloc[0], time_sorted.iloc[train_end-1]),
            'val_dates': (time_sorted.iloc[train_end], time_sorted.iloc[val_end-1]),
            'test_dates': (time_sorted.iloc[val_end], time_sorted.iloc[test_end-1]),
            'oot_dates': (time_sorted.iloc[test_end], time_sorted.iloc[-1])
        }

        # Log split information
        logger.info(f"Total samples: {n}")
        logger.info(f"Train: {len(splits['train_idx'])} samples ({train_ratio*100:.0f}%) | "
                   f"{splits['train_dates'][0]} to {splits['train_dates'][1]}")
        logger.info(f"Val:   {len(splits['val_idx'])} samples ({val_ratio*100:.0f}%) | "
                   f"{splits['val_dates'][0]} to {splits['val_dates'][1]}")
        logger.info(f"Test:  {len(splits['test_idx'])} samples ({test_ratio*100:.0f}%) | "
                   f"{splits['test_dates'][0]} to {splits['test_dates'][1]}")
        logger.info(f"OOT:   {len(splits['oot_idx'])} samples ({oot_ratio*100:.0f}%) | "
                   f"{splits['oot_dates'][0]} to {splits['oot_dates'][1]}")
        logger.info("="*80 + "\n")

        return splits

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_index: pd.Series,
        feature_names: list
    ) -> Tuple[xgb.XGBModel, Dict]:
        """
        Train XGBoost model with proper train/val/test/OOT splits.

        Returns
        -------
        model : xgb.XGBModel
            Trained model
        metrics : dict
            Training and evaluation metrics on all splits
        """
        logger.info("Training XGBoost model...")

        # Create temporal splits
        splits = self.split_train_val_test_oot(X, y, time_index)

        # Extract split data
        X_train = X.loc[splits['train_idx']]
        y_train = y.loc[splits['train_idx']]

        X_val = X.loc[splits['val_idx']]
        y_val = y.loc[splits['val_idx']]

        X_test = X.loc[splits['test_idx']]
        y_test = y.loc[splits['test_idx']]

        X_oot = X.loc[splits['oot_idx']]
        y_oot = y.loc[splits['oot_idx']]

        # Combine train+val for cross-validation and final training
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        # Time series split (respects temporal order) - only on train+val
        tscv = TimeSeriesSplit(n_splits=5)

        # Initialize model
        if self.task == "classification":
            model = xgb.XGBClassifier(**self.xgb_params["classification"])
        else:
            model = xgb.XGBRegressor(**self.xgb_params["regression"])

        # Cross-validation on train+val
        logger.info("Performing time series cross-validation on train+val...")
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val), 1):
            X_cv_train = X_train_val.iloc[train_idx]
            y_cv_train = y_train_val.iloc[train_idx]
            X_cv_val = X_train_val.iloc[val_idx]
            y_cv_val = y_train_val.iloc[val_idx]

            # Train on this fold
            model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)

            # Evaluate
            if self.task == "classification":
                y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
                score = roc_auc_score(y_cv_val, y_pred_proba)
                logger.info(f"  Fold {fold} AUC: {score:.4f}")
            else:
                y_pred = model.predict(X_cv_val)
                score = np.sqrt(np.mean((y_cv_val - y_pred) ** 2))
                logger.info(f"  Fold {fold} RMSE: {score:.4f}")

            cv_scores.append(score)

        logger.info(f"CV Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Train final model on train+val only (NOT entire dataset!)
        logger.info("Training final model on train+val...")
        model.fit(X_train_val, y_train_val, verbose=False)

        # Evaluate on all splits
        logger.info("\n" + "="*80)
        logger.info("EVALUATION ON ALL SPLITS")
        logger.info("="*80)

        train_metrics = self._evaluate_model(model, X_train, y_train, prefix="train")
        logger.info(f"Train: {self._format_metrics(train_metrics)}")

        val_metrics = self._evaluate_model(model, X_val, y_val, prefix="val")
        logger.info(f"Val:   {self._format_metrics(val_metrics)}")

        test_metrics = self._evaluate_model(model, X_test, y_test, prefix="test")
        logger.info(f"Test:  {self._format_metrics(test_metrics)}")

        oot_metrics = self._evaluate_model(model, X_oot, y_oot, prefix="oot")
        logger.info(f"OOT:   {self._format_metrics(oot_metrics)}")

        # Aggregate all metrics
        metrics = {
            'cv_scores': cv_scores,
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            **train_metrics,
            **val_metrics,
            **test_metrics,
            **oot_metrics
        }

        logger.info("="*80 + "\n")

        return model, metrics

    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics for logging."""
        if self.task == "classification":
            acc = metrics.get('accuracy', metrics.get('train_accuracy', metrics.get('val_accuracy',
                              metrics.get('test_accuracy', metrics.get('oot_accuracy', 0)))))
            auc = metrics.get('auc', metrics.get('train_auc', metrics.get('val_auc',
                              metrics.get('test_auc', metrics.get('oot_auc', 0)))))
            return f"Accuracy={acc:.4f}, AUC={auc:.4f}"
        else:
            rmse = metrics.get('rmse', metrics.get('train_rmse', metrics.get('val_rmse',
                               metrics.get('test_rmse', metrics.get('oot_rmse', 0)))))
            mae = metrics.get('mae', metrics.get('train_mae', metrics.get('val_mae',
                              metrics.get('test_mae', metrics.get('oot_mae', 0)))))
            return f"RMSE={rmse:.4f}, MAE={mae:.4f}"

    def _evaluate_model(self, model: xgb.XGBModel, X: pd.DataFrame, y: pd.Series, prefix: str = "") -> Dict:
        """Evaluate model and return metrics with optional prefix."""
        if self.task == "classification":
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]

            metrics = {
                f'{prefix}_accuracy' if prefix else 'accuracy': float(accuracy_score(y, y_pred)),
                f'{prefix}_auc' if prefix else 'auc': float(roc_auc_score(y, y_pred_proba))
            }

            # Only include detailed reports for test/oot (not train/val to save space)
            if prefix in ['test', 'oot', '']:
                metrics[f'{prefix}_confusion_matrix' if prefix else 'confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
                metrics[f'{prefix}_classification_report' if prefix else 'classification_report'] = classification_report(y, y_pred, output_dict=True)
        else:
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - y_pred))

            metrics = {
                f'{prefix}_mse' if prefix else 'mse': float(mse),
                f'{prefix}_rmse' if prefix else 'rmse': float(rmse),
                f'{prefix}_mae' if prefix else 'mae': float(mae),
                f'{prefix}_r2' if prefix else 'r2': float(model.score(X, y))
            }

        return metrics

    def save_model_and_artifacts(
        self,
        model: xgb.XGBModel,
        feature_names: list,
        metrics: Dict
    ):
        """Save trained model, metrics, and visualizations."""
        logger.info("Saving model and artifacts...")

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"xgboost_{self.task}_{self.prediction_horizon}min_{timestamp}"

        # Save model
        model_path = self.output_dir / f"{model_name}.pkl"
        dump(model, model_path)
        logger.info(f"  Model saved: {model_path}")

        # Save feature names
        feature_path = self.output_dir / f"{model_name}_features.json"
        with open(feature_path, 'w') as f:
            json.dump({'features': feature_names}, f, indent=2)

        # Save metrics
        metrics_path = self.output_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"  Metrics saved: {metrics_path}")

        # Save feature importance
        self._plot_feature_importance(model, feature_names, model_name)

        logger.info(f"  All artifacts saved to: {self.output_dir}")

    def _plot_feature_importance(
        self,
        model: xgb.XGBModel,
        feature_names: list,
        model_name: str
    ):
        """Plot and save feature importance."""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Save CSV
        importance_path = self.output_dir / f"{model_name}_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title(f'Top 20 Feature Importance - {model_name}')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()

        plot_path = self.output_dir / f"{model_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Feature importance plot: {plot_path}")

    def run(self):
        """Execute the full training pipeline."""
        logger.info("\n" + "="*80)
        logger.info(f"XGBoost Training Pipeline - {self.prediction_horizon}min Prediction")
        logger.info("="*80 + "\n")

        # Load data (market features + gold labels)
        market_df, news_df = self.load_data()

        # Validate labels are present
        market_df = self.validate_labels(market_df)

        # Merge with news
        combined_df = self.merge_market_news(market_df, news_df)

        # Prepare features
        X, y, time_index, feature_names = self.prepare_features(combined_df)

        # Train model with proper splits
        model, metrics = self.train_model(X, y, time_index, feature_names)

        # Save everything
        self.save_model_and_artifacts(model, feature_names, metrics)

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Task: {self.task}")
        logger.info(f"Prediction horizon: {self.prediction_horizon} minutes")
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Features: {len(feature_names)}")

        if self.task == "classification":
            logger.info(f"\nTest Set - Accuracy: {metrics.get('test_accuracy', 0):.4f}, "
                       f"AUC: {metrics.get('test_auc', 0):.4f}")
            logger.info(f"OOT Set  - Accuracy: {metrics.get('oot_accuracy', 0):.4f}, "
                       f"AUC: {metrics.get('oot_auc', 0):.4f}")
        else:
            logger.info(f"\nTest Set - RMSE: {metrics.get('test_rmse', 0):.4f}, "
                       f"MAE: {metrics.get('test_mae', 0):.4f}")
            logger.info(f"OOT Set  - RMSE: {metrics.get('oot_rmse', 0):.4f}, "
                       f"MAE: {metrics.get('oot_mae', 0):.4f}")

        logger.info(f"\nModel saved to: {self.output_dir}")
        logger.info("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-features",
        type=Path,
        required=True,
        help="Path to market features CSV (Gold layer)"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Path to labels CSV (Gold layer). If not provided, will be inferred from market features path."
    )
    parser.add_argument(
        "--news-signals",
        type=Path,
        help="Path to news signals CSV (Gold layer)"
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=30,
        help="Prediction horizon in minutes (default: 30)"
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Prediction task type"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_clean/models"),
        help="Output directory for models"
    )

    args = parser.parse_args()

    pipeline = XGBoostTrainingPipeline(
        market_features_path=args.market_features,
        labels_path=args.labels,
        news_signals_path=args.news_signals,
        prediction_horizon_minutes=args.prediction_horizon,
        output_dir=args.output_dir,
        task=args.task
    )

    pipeline.run()


if __name__ == "__main__":
    main()
