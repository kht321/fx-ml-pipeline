"""
XGBoost Training Pipeline - 30-Minute Price Prediction

Repository Location: fx-ml-pipeline/src_clean/training/xgboost_training_pipeline.py

Purpose:
    Trains XGBoost models to predict S&P 500 price movement in next 30 minutes.
    Uses both market features (Gold layer) and news signals for prediction.

Target Variable:
    - Binary classification: Price up/down in next 30 minutes
    - Regression: Actual price change in next 30 minutes

Features:
    - Market: Technical indicators, microstructure, volatility (37 features)
    - News: Sentiment signals, trading signals (11 features)
    - Total: ~48 features

Output:
    - Trained XGBoost model (pkl file)
    - Feature importance rankings
    - Evaluation metrics (accuracy, precision, recall, F1, AUC)
    - Training visualizations

Usage:
    python src_clean/training/xgboost_training_pipeline.py \\
        --market-features data_clean/gold/market/features/spx500_features.csv \\
        --news-signals data_clean/gold/news/signals/spx500_news_signals.csv \\
        --prediction-horizon 30 \\
        --output-dir data_clean/models
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
        news_signals_path: Optional[Path],
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
        """Load market features and news signals."""
        logger.info("Loading data...")

        # Load market features
        market_df = pd.read_csv(self.market_features_path)
        market_df['time'] = pd.to_datetime(market_df['time'])
        logger.info(f"Loaded market features: {len(market_df)} rows, {len(market_df.columns)} columns")

        # Load news signals if available
        news_df = None
        if self.news_signals_path and self.news_signals_path.exists():
            news_df = pd.read_csv(self.news_signals_path)
            news_df['signal_time'] = pd.to_datetime(news_df['signal_time'])
            logger.info(f"Loaded news signals: {len(news_df)} rows, {len(news_df.columns)} columns")
        else:
            logger.warning("No news signals provided - using market features only")

        return market_df, news_df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target labels for prediction.

        Label Types:
        - Classification: 1 if price up in next N minutes, 0 otherwise
        - Regression: Actual price change in next N minutes
        """
        logger.info(f"Creating labels for {self.prediction_horizon}-minute prediction...")

        df = df.sort_values('time').reset_index(drop=True)

        # Calculate future price (N minutes ahead)
        # For 1-minute data, shift by N rows
        df['future_close'] = df['close'].shift(-self.prediction_horizon)

        if self.task == "classification":
            # Binary: 1 if price goes up, 0 if down
            df['target'] = (df['future_close'] > df['close']).astype(int)
        else:
            # Regression: Actual price change
            df['target'] = df['future_close'] - df['close']

        # Remove rows without future prices (end of dataset)
        df = df.dropna(subset=['target'])

        logger.info(f"Created labels: {len(df)} valid samples")

        if self.task == "classification":
            value_counts = df['target'].value_counts()
            logger.info(f"Label distribution: Up={value_counts.get(1, 0)}, Down={value_counts.get(0, 0)}")

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

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepare features for training.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        feature_names : list
            List of feature names
        """
        logger.info("Preparing features...")

        # Columns to exclude from features
        exclude_cols = {
            'time', 'instrument', 'granularity',
            'target', 'future_close', 'signal_time',
            'collected_at', 'event_timestamp'
        }

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy()
        y = df['target'].copy()

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

        return X, y, list(X.columns)

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list
    ) -> Tuple[xgb.XGBModel, Dict]:
        """
        Train XGBoost model with time series cross-validation.

        Returns
        -------
        model : xgb.XGBModel
            Trained model
        metrics : dict
            Training and evaluation metrics
        """
        logger.info("Training XGBoost model...")

        # Time series split (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=5)

        # Initialize model
        if self.task == "classification":
            model = xgb.XGBClassifier(**self.xgb_params["classification"])
        else:
            model = xgb.XGBRegressor(**self.xgb_params["regression"])

        # Cross-validation
        logger.info("Performing time series cross-validation...")
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train on this fold
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Evaluate
            if self.task == "classification":
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
                logger.info(f"  Fold {fold} AUC: {score:.4f}")
            else:
                y_pred = model.predict(X_val)
                score = np.sqrt(np.mean((y_val - y_pred) ** 2))
                logger.info(f"  Fold {fold} RMSE: {score:.4f}")

            cv_scores.append(score)

        logger.info(f"CV Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Train final model on all data
        logger.info("Training final model on full dataset...")
        model.fit(X, y, verbose=False)

        # Evaluate on training set (for reference)
        metrics = self._evaluate_model(model, X, y)
        metrics['cv_scores'] = cv_scores
        metrics['cv_mean'] = float(np.mean(cv_scores))
        metrics['cv_std'] = float(np.std(cv_scores))

        return model, metrics

    def _evaluate_model(self, model: xgb.XGBModel, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model and return metrics."""
        if self.task == "classification":
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]

            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'auc': float(roc_auc_score(y, y_pred_proba)),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'classification_report': classification_report(y, y_pred, output_dict=True)
            }
        else:
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - y_pred))

            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(model.score(X, y))
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

        # Load data
        market_df, news_df = self.load_data()

        # Create labels
        market_df = self.create_labels(market_df)

        # Merge with news
        combined_df = self.merge_market_news(market_df, news_df)

        # Prepare features
        X, y, feature_names = self.prepare_features(combined_df)

        # Train model
        model, metrics = self.train_model(X, y, feature_names)

        # Save everything
        self.save_model_and_artifacts(model, feature_names, metrics)

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Task: {self.task}")
        logger.info(f"Prediction horizon: {self.prediction_horizon} minutes")
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Features: {len(feature_names)}")

        if self.task == "classification":
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"AUC: {metrics['auc']:.4f}")
        else:
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")

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
        news_signals_path=args.news_signals,
        prediction_horizon_minutes=args.prediction_horizon,
        output_dir=args.output_dir,
        task=args.task
    )

    pipeline.run()


if __name__ == "__main__":
    main()
