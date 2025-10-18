"""
XGBoost Training Pipeline with MLflow Integration - 30-Minute Price Prediction

Enhanced version with:
- MLflow experiment tracking
- Hyperparameter tuning (optional)
- Model registry integration
- Better model versioning

Target Variable:
- Classification: Predicts direction (Up=1, Down=0)
- Regression: Predicts percentage returns (not absolute price difference)

Why percentage returns?
- Prevents naive persistence (model learning yt = yt-1)
- Stationary target variable
- Scale-independent (works across different price levels)
- Directly interpretable for trading (% gain/loss)

To convert regression predictions back to price:
    predicted_price = current_price * (1 + predicted_return)

Repository Location: fx-ml-pipeline/src_clean/training/xgboost_training_pipeline_mlflow.py
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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_recall_fscore_support, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import mlflow
import mlflow.xgboost
import mlflow.sklearn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostMLflowTrainingPipeline:
    """Training pipeline for XGBoost with MLflow tracking."""

    def __init__(
        self,
        market_features_path: Path,
        news_signals_path: Optional[Path],
        prediction_horizon_minutes: int = 30,
        output_dir: Path = Path("data_clean/models"),
        task: str = "classification",
        experiment_name: str = "sp500_prediction",
        enable_tuning: bool = False
    ):
        """
        Initialize training pipeline with MLflow.

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
        experiment_name : str
            MLflow experiment name
        enable_tuning : bool
            Enable hyperparameter tuning
        """
        self.market_features_path = market_features_path
        self.news_signals_path = news_signals_path
        self.prediction_horizon = prediction_horizon_minutes
        self.output_dir = output_dir
        self.task = task
        self.experiment_name = experiment_name
        self.enable_tuning = enable_tuning

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

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

        # Hyperparameter search space
        self.param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [100, 200, 300],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9]
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
        """Create target labels for prediction."""
        logger.info(f"Creating labels for {self.prediction_horizon}-minute prediction...")

        df = df.sort_values('time').reset_index(drop=True)

        # Calculate future price
        df['future_close'] = df['close'].shift(-self.prediction_horizon)

        if self.task == "classification":
            # Classification: predict direction (up/down)
            df['target'] = (df['future_close'] > df['close']).astype(int)
        else:
            # Regression: predict percentage returns (not absolute price difference)
            # This prevents the model from learning naive persistence (yt = yt-1)
            df['target'] = (df['future_close'] - df['close']) / df['close']
            logger.info(f"Predicting percentage returns (stationary target)")

        df = df.dropna(subset=['target'])

        logger.info(f"Created labels: {len(df)} valid samples")

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
        """Merge market features with news signals using as-of join."""
        if news_df is None or news_df.empty:
            logger.info("No news data - using market features only")
            return market_df

        logger.info("Merging market features with news signals...")

        merged_rows = []
        tolerance = pd.Timedelta(hours=tolerance_hours)

        market_df = market_df.sort_values('time')
        news_df = news_df.sort_values('signal_time')

        news_features = [
            'signal_time', 'avg_sentiment', 'signal_strength',
            'trading_signal', 'article_count', 'quality_score'
        ]
        available_news = [c for c in news_features if c in news_df.columns]

        for _, market_row in market_df.iterrows():
            market_time = market_row['time']
            news_cutoff = market_time - tolerance
            eligible_news = news_df[
                (news_df['signal_time'] <= market_time) &
                (news_df['signal_time'] >= news_cutoff)
            ]

            merged_row = market_row.to_dict()

            if not eligible_news.empty:
                latest_news = eligible_news.iloc[-1]
                for col in available_news:
                    if col != 'signal_time':
                        merged_row[f'news_{col}'] = latest_news[col]
                news_age_minutes = (market_time - latest_news['signal_time']).total_seconds() / 60
                merged_row['news_age_minutes'] = news_age_minutes
                merged_row['news_available'] = 1
            else:
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
        """Prepare features for training."""
        logger.info("Preparing features...")

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
        """Train XGBoost model with MLflow tracking."""
        logger.info("Training XGBoost model with MLflow tracking...")

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("prediction_horizon_minutes", self.prediction_horizon)
            mlflow.log_param("task_type", self.task)
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("news_available", 'news_available' in X.columns)

            # Time series split
            tscv = TimeSeriesSplit(n_splits=5)

            if self.enable_tuning:
                logger.info("Performing hyperparameter tuning...")
                model = self._tune_hyperparameters(X, y, tscv)
            else:
                # Use default parameters
                if self.task == "classification":
                    model = xgb.XGBClassifier(**self.xgb_params["classification"])
                else:
                    model = xgb.XGBRegressor(**self.xgb_params["regression"])

                # Log model parameters
                mlflow.log_params(self.xgb_params[self.task])

            # Cross-validation
            logger.info("Performing time series cross-validation...")
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                if self.task == "classification":
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred_proba)
                    logger.info(f"  Fold {fold} AUC: {score:.4f}")
                else:
                    y_pred = model.predict(X_val)
                    score = np.sqrt(np.mean((y_val - y_pred) ** 2))
                    logger.info(f"  Fold {fold} RMSE: {score:.4f}")

                cv_scores.append(score)
                mlflow.log_metric(f"cv_fold_{fold}_score", score)

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            logger.info(f"CV Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")

            mlflow.log_metric("cv_mean", cv_mean)
            mlflow.log_metric("cv_std", cv_std)

            # Train final model
            logger.info("Training final model on full dataset...")
            model.fit(X, y, verbose=False)

            # Evaluate
            metrics = self._evaluate_model(model, X, y)
            metrics['cv_scores'] = cv_scores
            metrics['cv_mean'] = float(cv_mean)
            metrics['cv_std'] = float(cv_std)

            # Log metrics to MLflow
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Log model to MLflow
            if self.task == "classification":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.xgboost.log_model(model, "model")

            # Log feature names
            mlflow.log_dict({"features": feature_names}, "features.json")

            # Save feature importance plot
            self._plot_feature_importance_mlflow(model, feature_names)

            logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

            return model, metrics

    def _tune_hyperparameters(self, X, y, tscv):
        """Perform hyperparameter tuning with GridSearchCV."""
        logger.info("Tuning hyperparameters...")

        if self.task == "classification":
            base_model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
                tree_method="hist"
            )
            scoring = 'roc_auc'
        else:
            base_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=42,
                tree_method="hist"
            )
            scoring = 'neg_root_mean_squared_error'

        grid_search = GridSearchCV(
            base_model,
            self.param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")

        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        return grid_search.best_estimator_

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

    def _plot_feature_importance_mlflow(self, model: xgb.XGBModel, feature_names: list):
        """Plot and log feature importance to MLflow."""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title(f'Top 20 Feature Importance - {self.task}')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()

        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()

        # Log importance CSV
        mlflow.log_dict(importance_df.to_dict('records'), "feature_importance.json")

    def save_model_and_artifacts(
        self,
        model: xgb.XGBModel,
        feature_names: list,
        metrics: Dict
    ):
        """Save trained model (in addition to MLflow)."""
        logger.info("Saving model artifacts locally...")

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

    def run(self):
        """Execute the full training pipeline with MLflow."""
        logger.info("\n" + "="*80)
        logger.info(f"XGBoost Training Pipeline with MLflow - {self.prediction_horizon}min Prediction")
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
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
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
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sp500_prediction",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--enable-tuning",
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI"
    )

    args = parser.parse_args()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_uri)

    pipeline = XGBoostMLflowTrainingPipeline(
        market_features_path=args.market_features,
        news_signals_path=args.news_signals,
        prediction_horizon_minutes=args.prediction_horizon,
        output_dir=args.output_dir,
        task=args.task,
        experiment_name=args.experiment_name,
        enable_tuning=args.enable_tuning
    )

    pipeline.run()


if __name__ == "__main__":
    main()
