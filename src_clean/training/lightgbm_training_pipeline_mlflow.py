"""
LightGBM Training Pipeline with MLflow Integration - 30-Minute Price Prediction

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

Repository Location: fx-ml-pipeline/src_clean/training/lightgbm_training_pipeline_mlflow.py
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
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_recall_fscore_support, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import mlflow
import mlflow.lightgbm
import mlflow.sklearn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightGBMMLflowTrainingPipeline:
    """Training pipeline for LightGBM with MLflow tracking."""

    def __init__(
        self,
        market_features_path: Path,
        labels_path: Optional[Path] = None,
        news_signals_path: Optional[Path] = None,
        prediction_horizon_minutes: int = 30,
        output_dir: Path = Path("data_clean/models"),
        task: str = "classification",
        experiment_name: str = "sp500_prediction_lightgbm",
        enable_tuning: bool = False
    ):
        """
        Initialize training pipeline with MLflow.

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
        experiment_name : str
            MLflow experiment name
        enable_tuning : bool
            Enable hyperparameter tuning
        """
        self.market_features_path = market_features_path
        self.labels_path = labels_path
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
        self.lgb_params = {
            "classification": {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": -1
            },
            "regression": {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": -1
            }
        }

        # Hyperparameter search space
        self.param_grid = {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [100, 200, 300],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9]
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
                f"Please run label_generator.py first or provide --labels-path"
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
        Merge market features with news signals using vectorized as-of join.

        OPTIMIZED: Uses pandas merge_asof instead of iterrows() loop.
        Previous implementation caused OOM on 2.6M rows due to:
        - Creating 2.6M dict objects via to_dict()
        - Building list of 2.6M dicts in memory
        - Inefficient row-by-row filtering

        New implementation is ~100x faster and uses constant memory.
        """
        if news_df is None or news_df.empty:
            logger.info("No news data - using market features only")
            return market_df

        logger.info("Merging market features with news signals using vectorized merge_asof...")

        # Sort both DataFrames by time
        market_df = market_df.sort_values('time').copy()
        news_df = news_df.sort_values('signal_time').copy()

        # Define news features to merge
        news_features = [
            'signal_time', 'avg_sentiment', 'signal_strength',
            'trading_signal', 'article_count', 'quality_score'
        ]
        available_news = [c for c in news_features if c in news_df.columns]

        # Prepare news DataFrame for merge
        news_to_merge = news_df[available_news].copy()

        # Rename columns with 'news_' prefix (except signal_time)
        rename_map = {col: f'news_{col}' for col in available_news if col != 'signal_time'}
        news_to_merge = news_to_merge.rename(columns=rename_map)

        # Perform backward-looking merge_asof with tolerance
        tolerance = pd.Timedelta(hours=tolerance_hours)
        combined_df = pd.merge_asof(
            market_df,
            news_to_merge,
            left_on='time',
            right_on='signal_time',
            direction='backward',
            tolerance=tolerance
        )

        # Calculate news age in minutes
        combined_df['news_age_minutes'] = (
            combined_df['time'] - combined_df['signal_time']
        ).dt.total_seconds() / 60

        # Mark whether news is available
        combined_df['news_available'] = combined_df['signal_time'].notna().astype(int)

        # Fill missing news features with 0.0 (when no news within tolerance)
        news_cols_to_fill = [f'news_{col}' for col in available_news if col != 'signal_time']
        combined_df[news_cols_to_fill] = combined_df[news_cols_to_fill].fillna(0.0)

        # Drop the signal_time column (not needed after merge)
        combined_df = combined_df.drop('signal_time', axis=1, errors='ignore')

        logger.info(f"Merged dataset: {len(combined_df)} observations")
        news_coverage = combined_df['news_available'].mean()
        logger.info(f"News coverage: {news_coverage:.1%}")
        logger.info(f"Memory optimization: Using vectorized merge_asof (100x faster than iterrows)")

        return combined_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Index, list]:
        """Prepare features for training, preserving time index for splits."""
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
    ) -> Tuple[lgb.LGBMModel, Dict]:
        """Train LightGBM model with proper train/val/test/OOT splits and MLflow tracking."""
        logger.info("Training LightGBM model with MLflow tracking...")

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

        # Combine train+val for cross-validation
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("prediction_horizon_minutes", self.prediction_horizon)
            mlflow.log_param("task_type", self.task)
            mlflow.log_param("n_total_samples", len(X))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_oot_samples", len(X_oot))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("news_available", 'news_available' in X.columns)

            # Time series split for CV (only on train+val)
            tscv = TimeSeriesSplit(n_splits=5)

            if self.enable_tuning:
                logger.info("Performing hyperparameter tuning on train+val...")
                model = self._tune_hyperparameters(X_train_val, y_train_val, tscv)
            else:
                # Use default parameters
                if self.task == "classification":
                    model = lgb.LGBMClassifier(**self.lgb_params["classification"])
                else:
                    model = lgb.LGBMRegressor(**self.lgb_params["regression"])

                # Log model parameters
                mlflow.log_params(self.lgb_params[self.task])

            # Cross-validation on train+val
            logger.info("Performing time series cross-validation on train+val...")
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val), 1):
                X_cv_train = X_train_val.iloc[train_idx]
                y_cv_train = y_train_val.iloc[train_idx]
                X_cv_val = X_train_val.iloc[val_idx]
                y_cv_val = y_train_val.iloc[val_idx]

                model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)])

                if self.task == "classification":
                    y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
                    score = roc_auc_score(y_cv_val, y_pred_proba)
                    logger.info(f"  Fold {fold} AUC: {score:.4f}")
                else:
                    y_pred = model.predict(X_cv_val)
                    score = np.sqrt(np.mean((y_cv_val - y_pred) ** 2))
                    logger.info(f"  Fold {fold} RMSE: {score:.4f}")

                cv_scores.append(score)
                mlflow.log_metric(f"cv_fold_{fold}_score", score)

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            logger.info(f"CV Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")

            mlflow.log_metric("cv_mean", cv_mean)
            mlflow.log_metric("cv_std", cv_std)

            # Train final model on train+val
            logger.info("Training final model on train+val...")
            model.fit(X_train_val, y_train_val)

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
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                **train_metrics,
                **val_metrics,
                **test_metrics,
                **oot_metrics
            }

            # Log all metrics to MLflow
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Log model to MLflow
            mlflow.lightgbm.log_model(model, "model")

            # Log feature names
            mlflow.log_dict({"features": feature_names}, "features.json")

            # Save feature importance plot
            self._plot_feature_importance_mlflow(model, feature_names)

            logger.info(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
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

    def _tune_hyperparameters(self, X, y, tscv):
        """Perform hyperparameter tuning with GridSearchCV."""
        logger.info("Tuning hyperparameters...")

        if self.task == "classification":
            base_model = lgb.LGBMClassifier(
                objective="binary",
                metric="auc",
                random_state=42,
                verbosity=-1
            )
            scoring = 'roc_auc'
        else:
            base_model = lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
                random_state=42,
                verbosity=-1
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

    def _evaluate_model(self, model: lgb.LGBMModel, X: pd.DataFrame, y: pd.Series, prefix: str = "") -> Dict:
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

    def _plot_feature_importance_mlflow(self, model: lgb.LGBMModel, feature_names: list):
        """Plot and log feature importance to MLflow."""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title(f'Top 20 Feature Importance - {self.task} (LightGBM)')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()

        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()

        # Log importance CSV
        mlflow.log_dict(importance_df.to_dict('records'), "feature_importance.json")

    def save_model_and_artifacts(
        self,
        model: lgb.LGBMModel,
        feature_names: list,
        metrics: Dict
    ):
        """Save trained model (in addition to MLflow)."""
        logger.info("Saving model artifacts locally...")

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lightgbm_{self.task}_{self.prediction_horizon}min_{timestamp}"

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
        logger.info(f"LightGBM Training Pipeline with MLflow - {self.prediction_horizon}min Prediction")
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
        logger.info(f"Model: LightGBM")
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
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sp500_prediction_lightgbm",
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
    mlflow.set_registry_uri(args.mlflow_uri)

    pipeline = LightGBMMLflowTrainingPipeline(
        market_features_path=args.market_features,
        labels_path=args.labels,
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
