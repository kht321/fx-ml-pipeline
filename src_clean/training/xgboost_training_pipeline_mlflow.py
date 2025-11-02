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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, root_mean_squared_error, mean_absolute_error
)
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from joblib import dump
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import optuna
import shap


def bin_cutting(shap_totals: pd.Series) -> Dict[str, list[str]]:
    """Return cumulative percentile feature groups based on SHAP totals."""
    shap_totals = shap_totals.sort_values(ascending=False)
    total = shap_totals.sum()
    if total <= 0:
        logger.warning("SHAP totals sum to zero; percentile bins unavailable.")
        return {}

    cumprop = shap_totals.cumsum() / total
    edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1])
    labels = [f"{int(edges[i] * 100)}–{int(edges[i + 1] * 100)}%" for i in range(len(edges) - 1)]
    bin_assignments = pd.cut(cumprop, bins=edges, labels=labels, include_lowest=True)

    feature_bins = {
        label: shap_totals[bin_assignments == label].index.tolist()
        for label in bin_assignments.cat.categories
        if (bin_assignments == label).any()
    }

    keys = list(feature_bins.keys())
    for idx, key in enumerate(keys):
        if idx == 0:
            continue
        feature_bins[key] = feature_bins[keys[idx - 1]] + feature_bins[key]

    pop_groups = []
    for idx, (key, features) in enumerate(feature_bins.items(), 1):
        group_total = shap_totals[features].sum() / total if total else 0.0
        if len(features) < 3:
            pop_groups.append(key)
            logger.debug(
                "Dropping SHAP percentile bin %s (bin %d): %d features with %.4f share (<3 features).",
                key,
                idx,
                len(features),
                group_total,
            )
        else:
            logger.info(
                "Keeping SHAP percentile bin %s: %d features with %.4f cumulative share.",
                key,
                len(features),
                group_total,
            )

    for key in pop_groups:
        feature_bins.pop(key, None)

    return feature_bins

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
        labels_path: Optional[Path] = None,
        news_signals_path: Optional[Path] = None,
        prediction_horizon_minutes: int = 30,
        output_dir: Path = Path("data_clean/models"),
        task: str = "classification",
        experiment_name: str = "sp500_prediction",
        enable_tuning: bool = False,
        accelerate_dataset: Optional[bool] = False,
        stage1_n_trials: int=5,
        stage2_n_trials: int=5
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
        self.prediction_horizon_minutes = prediction_horizon_minutes  # for other class dependencies
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir / f"{experiment_name}_{run_timestamp}"
        self.task = task
        self.experiment_name = experiment_name
        self.enable_tuning = enable_tuning
        self.accelerate_dataset = accelerate_dataset

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hyperparameter_dir = self.output_dir / "hyperparameters" 
        self.stage1_n_trials = stage1_n_trials
        self.stage2_n_trials = stage2_n_trials

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
        """        self.param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [100, 200, 300],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9]
        }"""

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
            # data_clean/gold/market/features/spx500_features.csv -> data_clean/gold/market/labels/spx500_labels_30min.csv
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

    def merge_market_news(self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        tolerance_hours: int = 6
    ) -> pd.DataFrame:
        """Left-join market data with news signals, forward-filling missing news values."""
        market_sorted = market_df.sort_values('time').reset_index(drop=True).copy()
        market_sorted['time'] = pd.to_datetime(market_sorted['time'])

        if news_df is None or news_df.empty:
            logger.info("No news data - using market features only")
            market_sorted['news_rows_since_update'] = pd.NA
            market_sorted['news_available'] = 0
            market_sorted['news_age_minutes'] = np.nan
            return market_sorted

        logger.info(f"Merging market features with news signals. Market Feature Initial Shape: {market_sorted.shape}")

        news_sorted = news_df.sort_values('signal_time').reset_index(drop=True).copy()
        news_sorted['signal_time'] = pd.to_datetime(news_sorted['signal_time'])

        news_value_cols = [col for col in news_sorted.columns if col != 'signal_time']
        rename_map = {'signal_time': 'news_signal_time', **{col: f'news_{col}' for col in news_value_cols}}
        news_prefixed = news_sorted.rename(columns=rename_map)

        merge_kwargs = {
            "left_on": "time",
            "right_on": "news_signal_time",
            "direction": "backward",
        }
        if tolerance_hours is not None:
            merge_kwargs["tolerance"] = pd.Timedelta(hours=tolerance_hours)

        combined_df = pd.merge_asof(market_sorted, news_prefixed, **merge_kwargs)

        news_time = combined_df['news_signal_time']
        is_new_news = news_time.notna() & news_time.ne(news_time.shift())
        event_id = is_new_news.cumsum()
        rows_since = combined_df.groupby(event_id).cumcount().astype('Int64')
        rows_since = rows_since.where(event_id > 0, pd.NA)

        news_cols_to_ffill = [col for col in combined_df.columns if col.startswith('news_')]
        if news_cols_to_ffill:
            combined_df[news_cols_to_ffill] = combined_df[news_cols_to_ffill].ffill()

        combined_df['news_rows_since_update'] = rows_since
        combined_df['news_available'] = combined_df['news_signal_time'].notna().astype(int)

        if 'news_signal_time' in combined_df.columns:
            combined_df['news_signal_time'] = pd.to_datetime(combined_df['news_signal_time'])
            combined_df['news_age_minutes'] = (
                combined_df['time'] - combined_df['news_signal_time']
            ).dt.total_seconds() / 60

        logger.info(f"Merged dataset shape: {combined_df.shape}")
        news_coverage = combined_df['news_available'].mean()
        fresh_news = (combined_df['news_rows_since_update']==0).mean()
        logger.info(f"News coverage: {news_coverage:.1%}. Fresh news fraction: {fresh_news:.1%}")

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
            'prediction_horizon_minutes', 'label_generated_at', 'fold', 'news_signal_time'
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

        """
        2025-11-02 18:14:11,633 - INFO - Train: 1575039 samples (60%) | 2020-10-13 21:25:00+00:00 to 2023-10-12 16:03:00+00:00
        2025-11-02 18:14:11,633 - INFO - Val:   525013 samples (20%) | 2023-10-12 16:04:00+00:00 to 2024-10-11 06:16:00+00:00
        2025-11-02 18:14:11,633 - INFO - Test:  262506 samples (10%) | 2024-10-11 06:17:00+00:00 to 2025-04-11 13:22:00+00:00
        2025-11-02 18:14:11,633 - INFO - OOT:   262507 samples (10%) | 2025-04-11 13:23:00+00:00 to 2025-10-10 20:29:00+00:00
        """

        n = len(X_sorted)
        train_end = np.where(time_sorted==pd.to_datetime('2023-10-12 16:03:00+00:00'))[0][0]  #int(n * train_ratio)
        val_end = np.where(time_sorted==pd.to_datetime('2024-10-11 06:16:00+00:00'))[0][0]  #int(n * (train_ratio + val_ratio))
        test_end = np.where(time_sorted==pd.to_datetime('2025-04-11 13:22:00+00:00'))[0][0]  #int(n * (train_ratio + val_ratio + test_ratio))

        splits = {
            'train_idx': X_sorted.index[:train_end],
            'val_idx': X_sorted.index[train_end:val_end],
            'test_idx': X_sorted.index[val_end:test_end],
            'oot_idx': X_sorted.index[test_end:-10000],  # exclude 10000 data for OOT2
            'train_dates': (time_sorted.iloc[0], time_sorted.iloc[train_end-1]),
            'val_dates': (time_sorted.iloc[train_end], time_sorted.iloc[val_end-1]),
            'test_dates': (time_sorted.iloc[val_end], time_sorted.iloc[test_end-1]),
            'oot_dates': (time_sorted.iloc[test_end], time_sorted.iloc[-10000-1])
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
        """Train XGBoost model with proper train/val/test/OOT splits and MLflow tracking."""
        logger.info("Training XGBoost model with MLflow tracking...")

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
        with mlflow.start_run(
            run_name=f"{self.experiment_name}_XGB_{datetime.now():%Y%m%d_%H%M%S}"
        ):
            # Log parameters
            mlflow.log_param("prediction_horizon_minutes", self.prediction_horizon)
            mlflow.log_param("task_type", self.task)
            mlflow.log_param("n_total_samples", len(X))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_oot_samples", len(X_oot))
            mlflow.log_param("n_total_features", len(feature_names))
            mlflow.log_param("news_available", 'news_available' in X.columns)

            # Time series split for CV (only on train+val)
            tscv = TimeSeriesSplit(n_splits=5)
            test_size = int(0.15 * X_train_val.shape[0])
            tscv = TimeSeriesSplit(n_splits=3, test_size=test_size)

            selected_feature_names = feature_names

            if self.enable_tuning:
                logger.info("Performing hyperparameter tuning on train+val...")
                model = self._tune_hyperparameters(X_train_val, y_train_val, tscv)
                selected_feature_names = getattr(self, "selected_feature_names", feature_names)
            else:
                # Use default parameters
                if self.task == "classification":
                    model = xgb.XGBClassifier(**self.xgb_params["classification"])
                else:
                    model = xgb.XGBRegressor(**self.xgb_params["regression"])

                # Log model parameters
                mlflow.log_params(self.xgb_params[self.task])
                self.selected_feature_names = feature_names
                self.selected_feature_group = "all"
                mlflow.log_param("selected_feature_group", "all")
                mlflow.log_param("n_selected_features", len(selected_feature_names))

            # Cross-validation on train+val
            logger.info("Performing time series cross-validation on train+val...")
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val), 1):
                X_cv_train = X_train_val.iloc[train_idx][selected_feature_names]
                y_cv_train = y_train_val.iloc[train_idx]
                X_cv_val = X_train_val.iloc[val_idx][selected_feature_names]
                y_cv_val = y_train_val.iloc[val_idx]

                model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)

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
            model.fit(X_train_val[selected_feature_names], y_train_val, verbose=False)

            # Evaluate on all splits
            logger.info("\n" + "="*80)
            logger.info("EVALUATION ON ALL SPLITS")
            logger.info("="*80)

            X_train_sel = X_train[selected_feature_names]
            X_val_sel = X_val[selected_feature_names]
            X_test_sel = X_test[selected_feature_names]
            X_oot_sel = X_oot[selected_feature_names]

            train_metrics, y_train_pred = self._evaluate_model(model, X_train_sel, y_train, prefix="train")
            logger.info(f"Train: {self._format_metrics(train_metrics)}")

            val_metrics, y_val_pred = self._evaluate_model(model, X_val_sel, y_val, prefix="val")
            logger.info(f"Val:   {self._format_metrics(val_metrics)}")

            test_metrics, y_test_pred = self._evaluate_model(model, X_test_sel, y_test, prefix="test")
            logger.info(f"Test:  {self._format_metrics(test_metrics)}")

            oot_metrics, y_oot_pred = self._evaluate_model(model, X_oot_sel, y_oot, prefix="oot")
            logger.info(f"OOT:   {self._format_metrics(oot_metrics)}")

            # ===================================================================
            # 1. Compile predictions dataframe
            # ===================================================================
            predictions_list = []
            
            # Train split
            for timestamp, pred, actual in zip(time_index.loc[splits["train_idx"]], y_train_pred, y_train):
                predictions_list.append({
                    'timestamp': timestamp,
                    'split': 'train',
                    'prediction': pred,
                    'actual': actual
                })
            
            # Validation split
            for timestamp, pred, actual in zip(time_index.loc[splits["val_idx"]], y_val_pred, y_val):
                predictions_list.append({
                    'timestamp': timestamp,
                    'split': 'val',
                    'prediction': pred,
                    'actual': actual
                })
            
            # Test split
            for timestamp, pred, actual in zip(time_index.loc[splits["test_idx"]], y_test_pred, y_test):
                predictions_list.append({
                    'timestamp': timestamp,
                    'split': 'test',
                    'prediction': pred,
                    'actual': actual
                })
            
            # OOT split
            for timestamp, pred, actual in zip(time_index.loc[splits["oot_idx"]], y_oot_pred, y_oot):
                predictions_list.append({
                    'timestamp': timestamp,
                    'split': 'oot',
                    'prediction': pred,
                    'actual': actual
                })
            
            predictions_df = pd.DataFrame(predictions_list)
            logger.info(f"Compiled {len(predictions_df)} predictions across all splits")
            
            # ===================================================================
            # 2. Save predictions to CSV
            # ===================================================================
            predictions_path = Path(self.output_dir /  self.experiment_name)
            predictions_path.mkdir(parents=True, exist_ok=True)
            predictions_path = predictions_path / "predictions_vs_actuals.csv"
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Saved predictions to {predictions_path}")
            
            # ===================================================================
            # 3. Create and save time-trend plot
            # ===================================================================
            # Set academic style
            plt.style.use('seaborn-v0_8-whitegrid')
            mpl.rcParams.update({
                'figure.dpi': 160,
                'savefig.dpi': 160,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.labelsize': 11,
                'axes.titlesize': 13,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.frameon': False,
                'grid.alpha': 0.25,
            })
            
            fig, ax = plt.subplots(figsize=(14, 5), layout='constrained')
            
            # Plot actuals and predictions
            ax.plot(predictions_df['timestamp'], predictions_df['actual'], 
                    label='Actual', color='#264653', linewidth=1.5, alpha=0.9)
            ax.plot(predictions_df['timestamp'], predictions_df['prediction'], 
                    label='Predicted', color='#e76f51', linewidth=1.5, alpha=0.8, 
                    linestyle='--')
            
            # Define split colors
            split_colors = {
                'train': '#2a9d8f',
                'val': '#e9c46a', 
                'test': '#f4a261',
                'oot': '#e76f51'
            }
            
            # Add vertical lines and labels for split boundaries
            for split_name in ['train', 'val', 'test', 'oot']:
                split_data = predictions_df[predictions_df['split'] == split_name]
                
                if not split_data.empty:
                    boundary = split_data['timestamp'].iloc[0]
                    
                    # Vertical line at split boundary
                    ax.axvline(boundary, color=split_colors[split_name], 
                              linestyle='--', linewidth=1.5, alpha=0.5)
                    
                    # Add split label at top
                    y_pos = ax.get_ylim()[1] * 0.95
                    ax.text(boundary, y_pos, f'  {split_name.upper()}',
                           rotation=0, verticalalignment='top',
                           fontsize=10, fontweight='bold',
                           color=split_colors[split_name],
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', 
                                    edgecolor=split_colors[split_name],
                                    alpha=0.8))
            
            # Calculate and display metrics per split
            metrics_text = []
            for split_name in ['train', 'val', 'test', 'oot']:
                split_data = predictions_df[predictions_df['split'] == split_name]
                if not split_data.empty:
                    mae = mean_absolute_error(split_data['actual'], split_data['prediction'])
                    rmse = root_mean_squared_error(split_data['actual'], split_data['prediction'])
                    metrics_text.append(f"{split_name.upper()}: MAE={mae:.4f}, RMSE={rmse:.4f}")
            
            # Add metrics box
            metrics_str = '\n'.join(metrics_text)
            ax.text(0.02, 0.02, metrics_str,
                   transform=ax.transAxes,
                   verticalalignment='bottom',
                   fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Labels and title
            ax.set_xlabel('Timestamp', fontsize=11)
            ax.set_ylabel(f'Return ({self.prediction_horizon_minutes}min)', fontsize=11)
            ax.set_title(f'Predictions vs Actuals - XGBoost Model', 
                        fontsize=13, fontweight='bold', pad=15)
            ax.legend(loc='upper right', fontsize=10)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Save figure
            plot_path = Path(predictions_path.parent / "predictions_vs_actuals.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=160)
            plt.close()

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
            metrics['selected_features'] = selected_feature_names
            metrics['selected_feature_group'] = getattr(self, "selected_feature_group", "all")

            # Log all metrics to MLflow
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Log model to MLflow
            mlflow.xgboost.log_model(model, "model")

            # Log feature names
            mlflow.log_dict({"features": selected_feature_names}, "features.json")

            # Save feature importance plot
            self._plot_feature_importance_mlflow(model, selected_feature_names)

            logger.info(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
            logger.info("="*80 + "\n")

            self.selected_feature_names = selected_feature_names
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
        """Run two-stage Optuna hyperparameter tuning with SHAP-based feature selection."""
        logger.info("Tuning hyperparameters (two-stage Optuna + SHAP)...")

        X_df = X.copy()
        y_series = y.copy()

        stage1_trials = getattr(self, "stage1_n_trials", 5)
        stage2_trials = getattr(self, "stage2_n_trials", 5)

        # Ensure hyperparameter artifact directory exists
        hyper_dir = getattr(self, "hyperparameter_dir", self.output_dir / "hyperparameters" / self.experiment_name)
        hyper_dir.mkdir(parents=True, exist_ok=True)

        ModelClass = xgb.XGBClassifier if self.task == "classification" else xgb.XGBRegressor
        direction = "maximize" if self.task == "classification" else "minimize"
        metric_name = "auc" if self.task == "classification" else "rmse"

        base_params = self.xgb_params[self.task].copy()

        # Build Optuna-compatible search space from legacy grid
        #grid = self.param_grid

        def _suggest_params(trial: optuna.Trial) -> Dict[str, float]:
            lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)

            # Conditional upper bound for n_estimators based on lr
            if lr < 0.07:
                n_estimators_hi = 3000*3
                n_estimators_lo = 600
            elif lr < 0.2:
                n_estimators_hi = 1500*3
                n_estimators_lo = 300
            else:
                n_estimators_hi = 600*3
                n_estimators_lo = 100
            
            params = {
            # learning capacity / complexity
            "learning_rate": lr,
            "n_estimators": trial.suggest_int("n_estimators", n_estimators_lo, n_estimators_hi),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),  # minimum split loss

            # sampling / column subsampling (regularises interaction structure)
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

            # L1 / L2-style penalties
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 20.0, log=True),
            }
            return params

        def _cross_val_score(params: Dict[str, float], columns: list[str]) -> float:
            scores = []
            X_subset = X_df.loc[:, columns]
            for train_idx, val_idx in tscv.split(X_subset, y_series):
                model_params = base_params | params
                model = ModelClass(**model_params)
                X_train = X_subset.iloc[train_idx]
                X_val = X_subset.iloc[val_idx]
                y_train = y_series.iloc[train_idx]
                y_val = y_series.iloc[val_idx]

                model.fit(X_train, y_train, verbose=False)

                if self.task == "classification":
                    proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, proba)
                else:
                    preds = model.predict(X_val)
                    score = np.sqrt(np.mean((y_val - preds) ** 2))
                scores.append(float(score))

            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=42)

        def _stage1_objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial)
            score = _cross_val_score(params, X_df.columns.tolist())
            return score

        stage1_name = f"{self.experiment_name}_stage1_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        stage1_study = optuna.create_study(direction=direction, sampler=sampler, study_name=stage1_name)
        stage1_study.optimize(_stage1_objective, n_trials=stage1_trials, show_progress_bar=False)

        stage1_best_params = stage1_study.best_trial.params
        stage1_score = stage1_study.best_value
        logger.info("Stage 1 complete: best %s=%.5f with params=%s", metric_name, stage1_score, stage1_best_params)
        mlflow.log_dict(
            stage1_best_params,
            f"hyperparameters/{self.experiment_name}/stage1_best_params.json"
        )

        # Train best stage 1 model to compute SHAP importances
        stage1_model_params = base_params | stage1_best_params
        stage1_model = ModelClass(**stage1_model_params)
        stage1_model.fit(X_df, y_series, verbose=False)

        explainer = shap.TreeExplainer(stage1_model)
        shap_values = explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            shap_array = np.stack(shap_values, axis=0).mean(axis=0)
        else:
            shap_array = np.array(shap_values)

        shap_abs = np.abs(shap_array)
        shap_totals = pd.Series(shap_abs.sum(axis=0), index=X_df.columns).sort_values(ascending=False)
        total_importance = shap_totals.sum()
        cumprop = shap_totals.cumsum() / total_importance if total_importance else shap_totals.cumsum()

        feature_groups: Dict[str, list[str]] = bin_cutting(shap_totals)
        if not feature_groups:
            feature_groups = {}

        straight_line = np.linspace(1 / len(shap_totals), 1.0, len(shap_totals))
        deviation = cumprop.values - straight_line
        knee_idx = int(np.argmax(deviation))
        knee_features = shap_totals.index[:knee_idx + 1].tolist()
        feature_groups["knee"] = knee_features or shap_totals.index[:1].tolist()

        # Stage 2: include feature group selection
        feature_group_keys = list(feature_groups.keys())

        def _stage2_objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial)
            group = trial.suggest_categorical("feature_group", feature_group_keys + ["all"])
            columns = X_df.columns.tolist() if group == "all" else feature_groups[group]
            if not columns:
                raise optuna.TrialPruned(f"Feature group '{group}' produced no columns.")
            trial.set_user_attr("feature_group", group)
            score = _cross_val_score(params, columns)
            return score

        stage2_name = f"{self.experiment_name}_stage2_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        stage2_study = optuna.create_study(direction=direction, sampler=sampler, study_name=stage2_name)
        # Make sure Stage 1 solution is evaluated in Stage 2 with all features
        stage2_study.enqueue_trial(stage1_best_params | {"feature_group": "all"})
        stage2_study.optimize(_stage2_objective, n_trials=stage2_trials, show_progress_bar=False)

        best_stage2_params = stage2_study.best_trial.params.copy()
        best_group = best_stage2_params.pop("feature_group", "all")
        selected_features = X_df.columns.tolist() if best_group == "all" else feature_groups.get(best_group, X_df.columns.tolist())
        mlflow.log_dict(
            {**best_stage2_params, "feature_group": best_group},
            f"hyperparameters/{self.experiment_name}/stage2_best_params.json"
        )

        final_params = base_params | best_stage2_params
        final_model = ModelClass(**final_params)

        self.selected_feature_names = selected_features
        self.selected_features = selected_features
        self.selected_feature_group = best_group
        self.stage1_study = stage1_study
        self.stage2_study = stage2_study

        # Persist artifacts
        timestamp = datetime.now().isoformat()
        shap_payload = {
            "generated_at": timestamp,
            "total_importance": float(total_importance),
            "knee_index": knee_idx,
            "knee_share": float(cumprop.iloc[knee_idx]) if len(cumprop) else 0.0,
            "feature_groups": feature_groups,
            "shap_totals": {col: float(val) for col, val in shap_totals.items()},
        }
        (hyper_dir / "shap_summary.json").write_text(json.dumps(shap_payload, indent=2), encoding="utf-8")

        shap_fig_path = hyper_dir / "shap_feature_importance.png"
        plt.figure(figsize=(12, 6))
        sns.barplot(x=shap_totals.values[:20], y=shap_totals.index[:20], orient="h", color="steelblue")
        plt.title("Top 20 SHAP Feature Importance")
        plt.xlabel("Total |SHAP|")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(shap_fig_path, dpi=180)
        plt.close()

        def _write_study_artifacts(study: optuna.Study, stage: str) -> None:
            trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            trials_path = hyper_dir / f"{stage}_trials.csv"
            trials_df.to_csv(trials_path, index=False)
            summary = {
                "generated_at": timestamp,
                "stage": stage,
                "direction": direction,
                "best_value": float(study.best_value),
                "best_params": study.best_trial.params,
                "n_trials": len(study.trials),
            }
            (hyper_dir / f"{stage}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            mlflow.log_table(
                trials_df,
                artifact_file=f"hyperparameters/{self.experiment_name}/{stage}_trials.json"
            )
            mlflow.log_dict(
                summary,
                f"hyperparameters/{self.experiment_name}/{stage}_summary.json"
            )

        _write_study_artifacts(stage1_study, "stage1")
        _write_study_artifacts(stage2_study, "stage2")

        mlflow.log_metric("stage1_best_score", float(stage1_score))
        mlflow.log_metric("stage2_best_score", float(stage2_study.best_value))
        mlflow.log_params({k: v for k, v in final_params.items() if k not in {"objective", "eval_metric"}})
        mlflow.log_param("selected_feature_group", best_group)
        mlflow.log_param("n_selected_features", len(selected_features))

        mlflow.log_artifacts(hyper_dir.as_posix(), artifact_path=f"hyperparameters/{self.experiment_name}")

        self.tuned_params_stage2 = final_params
        return final_model

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

        return metrics, y_pred

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

        # Load data (market features + gold labels)
        market_df, news_df = self.load_data()

        # Validate labels are present
        market_df = self.validate_labels(market_df)
        if self.accelerate_dataset:
            market_df = market_df.head(10000)
            logger.info("Dataset accelerated for TESTING ENVIRONMENT: using first 10,000 samples for training.")

        # Merge with news
        combined_df = self.merge_market_news(market_df, news_df)

        # Prepare features
        X, y, time_index, feature_names = self.prepare_features(combined_df)

        # Train model with proper splits
        model, metrics = self.train_model(X, y, time_index, feature_names)
        selected_feature_names = metrics.get('selected_features', feature_names)

        # Save everything
        self.save_model_and_artifacts(model, selected_feature_names, metrics)

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Task: {self.task}")
        logger.info(f"Prediction horizon: {self.prediction_horizon} minutes")
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Features: {len(selected_feature_names)}")

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
    mlflow.set_registry_uri(args.mlflow_uri)

    pipeline = XGBoostMLflowTrainingPipeline(
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
