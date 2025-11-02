"""
Autoregressive OLS Training Pipeline with MLflow Integration.

This pipeline extends the XGBoost MLflow training pipeline to train an
autoregressive (AR) model using statsmodels' OLS implementation. It keeps the
same data-loading and validation utilities but replaces feature engineering,
hyperparameter tuning, and model training with AR-specific logic.
"""

import argparse
import json
import logging
import math
import tempfile
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt

from .xgboost_training_pipeline_mlflow import XGBoostMLflowTrainingPipeline

# Configure logging consistent with other training pipelines.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutoregressiveOLSMLflowTrainingPipeline(XGBoostMLflowTrainingPipeline):
    """Training pipeline for autoregressive OLS models with MLflow tracking."""

    def __init__(
        self,
        market_features_path: Path,
        labels_path: Optional[Path] = None,
        news_signals_path: Optional[Path] = None,
        prediction_horizon_minutes: int = 30,
        output_dir: Path = Path("data_clean/models"),
        experiment_name: str = "sp500_prediction",
        lag_min: int = 1,
        lag_max: int = 7,
        enable_tuning: Optional[bool] = False,
        accelerate_dataset: Optional[bool] = False,
    ):
        """
        Initialize the autoregressive training pipeline.

        Parameters
        ----------
        market_features_path : Path
            Path to Gold layer market features CSV.
        labels_path : Path, optional
            Path to Gold layer labels CSV (inferred if omitted).
        news_signals_path : Path, optional
            Path to Gold layer news signals CSV (ignored for AR models).
        prediction_horizon_minutes : int
            Prediction horizon in minutes (default: 30).
        output_dir : Path
            Directory to store trained models and artifacts.
        experiment_name : str
            MLflow experiment name.
        lag_min : int
            Minimum number of lags to evaluate during tuning.
        lag_max : int
            Maximum number of lags to evaluate during tuning.
        """
        super().__init__(
            market_features_path=market_features_path,
            labels_path=labels_path,
            news_signals_path=news_signals_path,
            prediction_horizon_minutes=prediction_horizon_minutes,
            output_dir=output_dir,
            task="regression",
            experiment_name=experiment_name,
            enable_tuning=False,
        )

        if lag_min < 1:
            raise ValueError("lag_min must be at least 1.")
        if lag_max < lag_min:
            raise ValueError("lag_max must be greater than or equal to lag_min.")

        self.lag_min = lag_min
        self.lag_max = lag_max
        self.selected_lag: Optional[int] = None
        self.enable_tuning = enable_tuning
        self.accelerate_dataset = accelerate_dataset

    def prepare_features(
        self,
        df: pd.DataFrame,
        lags: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
        """
        Build lagged design matrix for the AR model.

        Parameters
        ----------
        df : pd.DataFrame
            Market dataframe containing `return_30` and `target` columns.
        lags : int, optional
            Number of lagged returns to include; defaults to `self.selected_lag`.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]
            Feature matrix with constant column, y_AR (t+1 return), y target (t+30 return),
            time index, and feature names.
        """
        lag_count = lags if lags is not None else self.selected_lag
        if lag_count is None:
            raise ValueError("Lag count is not set. Provide `lags` or run hyperparameter tuning first.")
        if lag_count < 1:
            raise ValueError("lags must be a positive integer.")

        for required_col in ("return_30", "target", "time"):
            if required_col not in df.columns:
                raise KeyError(f"Column '{required_col}' is required for autoregressive modelling.")

        df_sorted = df.sort_values("time").reset_index(drop=True)
        df_sorted["return_30"] = pd.to_numeric(df_sorted["return_30"], errors="coerce") * 100 # because target is in percent...
        df_sorted["target"] = pd.to_numeric(df_sorted["target"], errors="coerce")
        df_sorted["time"] = pd.to_datetime(df_sorted["time"])

        base = pd.DataFrame(
            {
                "return_30": df_sorted["return_30"],
                "target": df_sorted["target"],
                "time": df_sorted["time"],
            }
        )

        for lag in range(1, lag_count + 1):
            base[f"return_30_lag{lag}"] = base["return_30"].shift(lag)

        base["y_ar"] = base["return_30"].shift(-1)

        feature_cols = ["return_30"] + [f"return_30_lag{lag}" for lag in range(1, lag_count + 1)]
        required_cols = feature_cols + ["y_ar", "target"]
        base = base.dropna(subset=required_cols).reset_index(drop=True)

        if base.empty:
            raise ValueError(
                f"No samples remain after generating {lag_count} lags. "
                "Check data sufficiency or reduce lag count."
            )

        for col in required_cols:
            base[col] = pd.to_numeric(base[col], errors="coerce")

        base = base.dropna(subset=required_cols).reset_index(drop=True)

        X = sm.add_constant(base[feature_cols], has_constant="add")
        y_ar = base["y_ar"].astype(float)
        y_target = base["target"].astype(float)
        time_index = base["time"]

        logger.info(
            "Prepared autoregressive features with %d lags (%d samples, %d features).",
            lag_count,
            len(X),
            X.shape[1],
        )

        return X, y_ar, y_target, time_index, list(X.columns)

    def tune_hyperparameters(
        self,
        df: pd.DataFrame,
    ) -> Tuple[int, List[Dict[str, float]]]:
        """
        Evaluate validation RMSE across lag counts and select the best configuration.

        Returns
        -------
        Tuple[int, List[Dict[str, float]]]
            Best lag count and per-lag evaluation records.
        """
        logger.info(
            "Tuning lag hyperparameter: evaluating lags %d through %d.",
            self.lag_min,
            self.lag_max,
        )

        best_lag: Optional[int] = None
        best_rmse = math.inf
        tuning_results: List[Dict[str, float]] = []
        horizon_steps = max(int(self.prediction_horizon), 1)

        for lag in range(self.lag_min, self.lag_max + 1):
            try:
                X, y_ar, y_target, time_index, _ = self.prepare_features(df, lags=lag)
            except ValueError as exc:
                logger.warning("Skipping lag %d: %s", lag, exc)
                continue

            splits = self.split_train_val_test_oot(X, y_target, time_index)
            X_train = X.loc[splits["train_idx"]]
            y_train_ar = y_ar.loc[splits["train_idx"]]
            X_val = X.loc[splits["val_idx"]]
            y_val_target = y_target.loc[splits["val_idx"]]

            if len(X_val) == 0:
                logger.warning("Lag %d produced an empty validation set. Skipping.", lag)
                continue

            model = sm.OLS(y_train_ar, X_train).fit()
            val_target_pred = self._forecast_split_target(
                model,
                X_val,
                lag_count=lag,
                horizon=horizon_steps,
            )
            val_rmse = root_mean_squared_error(y_val_target, val_target_pred)
            val_mae = mean_absolute_error(y_val_target, val_target_pred)

            record = {
                "lag": lag,
                "val_rmse_target": float(val_rmse),
                "val_mae_target": float(val_mae),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "train_samples": int(model.nobs),
                "val_samples": int(len(X_val)),
            }
            tuning_results.append(record)

            mlflow.log_metric(f"val_rmse_target_lag_{lag}", val_rmse)
            mlflow.log_metric(f"val_mae_target_lag_{lag}", val_mae)
            mlflow.log_metric(f"aic_lag_{lag}", model.aic)
            mlflow.log_metric(f"bic_lag_{lag}", model.bic)

            logger.info(
                "Lag %d -> Validation RMSE@t+%d: %.6f | MAE: %.6f | AIC: %.2f | BIC: %.2f",
                lag,
                horizon_steps,
                val_rmse,
                val_mae,
                model.aic,
                model.bic,
            )

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_lag = lag

        if not tuning_results:
            raise RuntimeError(
                "Lag search failed: no configuration produced a valid train/validation split."
            )

        mlflow.log_metric("best_val_rmse_target", best_rmse)
        logger.info("Selected lag %d with validation RMSE %.6f.", best_lag, best_rmse)

        return best_lag, tuning_results

    def _log_metric_if_finite(self, name: str, value: float) -> None:
        """Log a metric to MLflow only if the value is finite."""
        if value is None:
            return
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(numeric_value):
            mlflow.log_metric(name, numeric_value)

    def _recursive_forecast_target(
        self,
        model,
        row: pd.Series,
        lag_count: int,
        horizon: int,
    ) -> Tuple[float, List[float]]:
        """Recursively forecast horizon steps ahead using 1-step AR model."""
        if horizon < 1:
            raise ValueError("Forecast horizon must be at least 1.")

        exog_names = [name for name in model.model.exog_names if name != "const"]
        const_value = float(model.params.get("const", 0.0))
        coef_values = model.params[exog_names].astype(float).to_numpy()

        history = [float(row["return_30"])]
        for lag_idx in range(1, lag_count + 1):
            col_name = f"return_30_lag{lag_idx}"
            if col_name not in row.index:
                raise KeyError(f"Missing column '{col_name}' in feature row for recursion.")
            history.append(float(row[col_name]))

        predictions: List[float] = []
        for _ in range(horizon):
            feature_values: List[float] = []
            for name in exog_names:
                if name == "return_30":
                    feature_values.append(history[0])
                elif name.startswith("return_30_lag"):
                    lag_idx = int(name.replace("return_30_lag", ""))
                    if lag_idx > lag_count:
                        raise ValueError(
                            f"Model expects lag {lag_idx}, but only {lag_count} lags prepared."
                        )
                    feature_values.append(history[lag_idx])
                else:
                    raise KeyError(f"Unexpected feature '{name}' in model exogenous names.")

            step_pred = const_value + float(np.dot(coef_values, np.array(feature_values, dtype=float)))
            predictions.append(step_pred)

            # Update history: newest prediction becomes "return_30", drop oldest lag.
            history = [step_pred] + history[:-1]

        return predictions[-1], predictions

    def _forecast_split_target(
        self,
        model,
        X_split: pd.DataFrame,
        lag_count: int,
        horizon: int,
        progress_bar: Optional[bool] = False,
    ) -> np.ndarray:
        """Generate horizon-step forecasts for each row in a split."""
        if X_split.empty:
            return np.array([], dtype=float)

        predictions: List[float] = []
        for _, row in tqdm(X_split.iterrows(), total=X_split.shape[0], disable=not progress_bar):
            forecast, _ = self._recursive_forecast_target(model, row, lag_count, horizon)
            predictions.append(forecast)

        return np.asarray(predictions, dtype=float)

    def train_model(
        self,
        df: pd.DataFrame,
    ):
        """
        Train the autoregressive model, logging all steps to MLflow.

        Returns
        -------
        Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict]
            Trained model and metrics dictionary.
        """
        logger.info("Training autoregressive OLS model with MLflow tracking...")

        horizon_steps = max(int(self.prediction_horizon), 1)
        if not os.getenv("MLFLOW_TRACKING_URI"):
            try:
                project_root = Path(__file__).resolve().parents[2]  # fx-ml-pipeline/
            except Exception:
                project_root = Path.cwd()
            tracking_dir = project_root / "mlruns"
            tracking_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_dir.as_uri())
        mlflow.set_experiment(self.experiment_name)
        logger.info("MLflow tracking URI: %s | Experiment: %s",
                    mlflow.get_tracking_uri(), self.experiment_name)

        with mlflow.start_run(
            run_name=f"{self.experiment_name}_AR_{datetime.now():%Y%m%d_%H%M%S}"
        ):
            mlflow.log_param("prediction_horizon_minutes", self.prediction_horizon)
            mlflow.log_param("task_type", "regression")
            mlflow.log_param("lag_search_min", self.lag_min)
            mlflow.log_param("lag_search_max", self.lag_max)
            mlflow.log_param("n_raw_samples", len(df))
            mlflow.log_param("forecast_horizon_steps", horizon_steps)
            if self.enable_tuning:
                best_lag, tuning_results = self.tune_hyperparameters(df)
            else:
                best_lag, tuning_results = 2, {}
            self.selected_lag = best_lag

            X, y_ar, y_target, time_index, feature_names = self.prepare_features(df, lags=best_lag)
            effective_samples = len(X)
            num_features = len(feature_names)

            mlflow.log_param("selected_lag", best_lag)
            mlflow.log_param("n_effective_samples", effective_samples)
            mlflow.log_param("n_features", num_features)
            mlflow.log_param("n_total_samples", len(X))

            splits = self.split_train_val_test_oot(X, y_target, time_index)
            X_train = X.loc[splits["train_idx"]]
            y_train_ar = y_ar.loc[splits["train_idx"]]
            y_train_target = y_target.loc[splits["train_idx"]]

            X_val = X.loc[splits["val_idx"]]
            y_val_ar = y_ar.loc[splits["val_idx"]]
            y_val_target = y_target.loc[splits["val_idx"]]

            X_test = X.loc[splits["test_idx"]]
            y_test_target = y_target.loc[splits["test_idx"]]

            X_oot = X.loc[splits["oot_idx"]]
            y_oot_target = y_target.loc[splits["oot_idx"]]

            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_oot_samples", len(X_oot))

            X_train_val = pd.concat([X_train, X_val])
            y_train_val_ar = pd.concat([y_train_ar, y_val_ar])
            logger.info(
                "Refitting final model on %d train and %d validation samples (total=%d).",
                len(X_train),
                len(X_val),
                len(X_train_val),
            )

            final_model = sm.OLS(y_train_val_ar, X_train_val).fit()
            logger.info(
                "Final model trained. R-squared: %.4f | Adj. R-squared: %.4f",
                final_model.rsquared,
                final_model.rsquared_adj,
            )

            train_pred_target = self._forecast_split_target(
                final_model,
                X_train,
                lag_count=best_lag,
                horizon=horizon_steps,
                progress_bar=True,
            )
            logger.info("Completed train predictions...")
            val_pred_target = self._forecast_split_target(
                final_model,
                X_val,
                lag_count=best_lag,
                horizon=horizon_steps,
                progress_bar=True,
            )
            logger.info("Completed validation predictions...")
            test_pred_target = self._forecast_split_target(
                final_model,
                X_test,
                lag_count=best_lag,
                horizon=horizon_steps,
                progress_bar=True,
            )
            logger.info("Completed test predictions...")
            oot_pred_target = self._forecast_split_target(
                final_model,
                X_oot,
                lag_count=best_lag,
                horizon=horizon_steps,
                progress_bar=True,
            )
            logger.info("Completed OOT predictions...")

            # ===================================================================
            # 1. Compile predictions dataframe
            # ===================================================================
            predictions_list = []
            
            # Train split
            for timestamp, pred, actual in zip(time_index.loc[splits["train_idx"]], train_pred_target, y_train_target):
                predictions_list.append({
                    'timestamp': timestamp,
                    'split': 'train',
                    'prediction': pred,
                    'actual': actual
                })
            
            # Validation split
            for timestamp, pred, actual in zip(time_index.loc[splits["val_idx"]], val_pred_target, y_val_target):
                predictions_list.append({
                    'timestamp': timestamp,
                    'split': 'val',
                    'prediction': pred,
                    'actual': actual
                })
            
            # Test split
            for timestamp, pred, actual in zip(time_index.loc[splits["test_idx"]], test_pred_target, y_test_target):
                predictions_list.append({
                    'timestamp': timestamp,
                    'split': 'test',
                    'prediction': pred,
                    'actual': actual
                })
            
            # OOT split
            for timestamp, pred, actual in zip(time_index.loc[splits["oot_idx"]], oot_pred_target, y_oot_target):
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
            ax.set_title(f'Predictions vs Actuals - AR({best_lag}) Model', 
                        fontsize=13, fontweight='bold', pad=15)
            ax.legend(loc='upper right', fontsize=10)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Save figure
            plot_path = Path(predictions_path.parent / "predictions_vs_actuals.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=160)
            plt.close()
            
            logger.info(f"Saved predictions plot to {plot_path}")
            
            # Log artifacts to MLflow
            mlflow.log_artifact(str(predictions_path))
            mlflow.log_artifact(str(plot_path))

            metrics: Dict[str, float] = {
                "selected_lag": int(best_lag),
                "train_rmse_t30": root_mean_squared_error(y_train_target, train_pred_target)
                if len(train_pred_target) > 0 else math.nan,
                "train_mae_t30": mean_absolute_error(y_train_target, train_pred_target)
                if len(train_pred_target) > 0 else math.nan,
                "val_rmse_t30": root_mean_squared_error(y_val_target, val_pred_target)
                if len(val_pred_target) > 0 else math.nan,
                "val_mae_t30": mean_absolute_error(y_val_target, val_pred_target)
                if len(val_pred_target) > 0 else math.nan,
                "test_rmse_t30": root_mean_squared_error(y_test_target, test_pred_target)
                if len(test_pred_target) > 0 else math.nan,
                "test_mae_t30": mean_absolute_error(y_test_target, test_pred_target)
                if len(test_pred_target) > 0 else math.nan,
                "oot_rmse_t30": root_mean_squared_error(y_oot_target, oot_pred_target)
                if len(oot_pred_target) > 0 else math.nan,
                "oot_mae_t30": mean_absolute_error(y_oot_target, oot_pred_target)
                if len(oot_pred_target) > 0 else math.nan,
                "model_rsquared": float(final_model.rsquared),
                "model_rsquared_adj": float(final_model.rsquared_adj),
            }

            logger.info(
                "Recursive evaluation (t+%d): "
                "Train RMSE=%.6f MAE=%.6f | "
                "Val RMSE=%.6f MAE=%.6f | "
                "Test RMSE=%.6f MAE=%.6f | "
                "OOT RMSE=%.6f MAE=%.6f",
                horizon_steps,
                metrics.get("train_rmse_t30", float("nan")),
                metrics.get("train_mae_t30", float("nan")),
                metrics.get("val_rmse_t30", float("nan")),
                metrics.get("val_mae_t30", float("nan")),
                metrics.get("test_rmse_t30", float("nan")),
                metrics.get("test_mae_t30", float("nan")),
                metrics.get("oot_rmse_t30", float("nan")),
                metrics.get("oot_mae_t30", float("nan")),
            )

            for metric_name, metric_value in metrics.items():
                if metric_name == "selected_lag":
                    continue
                self._log_metric_if_finite(metric_name, metric_value)

            coefficients = {name: float(value) for name, value in final_model.params.items()}
            mlflow.log_dict({"coefficients": coefficients}, "coefficients.json")
            mlflow.log_dict({"tuning_results": tuning_results}, "lag_tuning_results.json")
            mlflow.log_text(final_model.summary().as_text(), "model_summary.txt")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / "ar_ols_model.pkl"
                final_model.save(str(tmp_path))
                mlflow.log_artifact(str(tmp_path), artifact_path="models")

            active_run = mlflow.active_run()
            if active_run is not None:
                metrics["mlflow_run_id"] = active_run.info.run_id

        metrics["feature_names"] = feature_names
        metrics["tuning_results"] = tuning_results
        metrics["n_effective_samples"] = effective_samples
        metrics["n_features"] = num_features
        metrics["horizon_steps"] = horizon_steps

        return final_model, metrics

    def save_model_and_artifacts(
        self,
        model,
        feature_names: List[str],
        metrics: Dict,
    ) -> None:
        """Persist the trained model, feature names, and metrics locally."""
        logger.info("Saving autoregressive model artifacts locally...")

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ar_ols_{self.prediction_horizon}min_{timestamp}"

        model_path = self.output_dir / f"{model_name}.pkl"
        model.save(str(model_path))
        logger.info("  Model saved: %s", model_path)

        feature_path = self.output_dir / f"{model_name}_features.json"
        with open(feature_path, "w", encoding="utf-8") as feature_file:
            json.dump(
                {
                    "features": feature_names,
                    "selected_lag": metrics.get("selected_lag"),
                },
                feature_file,
                indent=2,
            )
        logger.info("  Feature metadata saved: %s", feature_path)

        metrics_path = self.output_dir / f"{model_name}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2, default=str)
        logger.info("  Metrics saved: %s", metrics_path)

    def run(self) -> None:
        """Execute the full AR training pipeline with MLflow integration."""
        logger.info("\n" + "=" * 80)
        logger.info("Autoregressive OLS Pipeline with MLflow - %smin Prediction", self.prediction_horizon)
        logger.info("=" * 80 + "\n")

        market_df, news_df = self.load_data()
        market_df = self.validate_labels(market_df)

        if news_df is not None and not news_df.empty:
            logger.info("News features provided but ignored for autoregressive modelling.")

        if self.accelerate_dataset:
            market_df = market_df.head(10000)
            logger.info("Dataset accelerated for TESTING ENVIRONMENT: using first 10,000 samples for training.")
        final_model, metrics = self.train_model(market_df)
        feature_names = metrics.get("feature_names", [])

        self.save_model_and_artifacts(final_model, feature_names, metrics)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info("Prediction horizon: %s minutes", self.prediction_horizon)
        logger.info("Forecast horizon (steps): %s", metrics.get("horizon_steps", "unknown"))
        logger.info("Selected lag order: %s", metrics.get("selected_lag"))
        logger.info(
            "Total samples (post-lagging): %s",
            metrics.get("n_effective_samples", "unknown"),
        )
        logger.info("Features: %s", metrics.get("n_features", "unknown"))
        logger.info(
            "Test Set  - RMSE: %.4f | MAE: %.4f",
            metrics.get("test_rmse_t30", float("nan")),
            metrics.get("test_mae_t30", float("nan")),
        )
        logger.info(
            "OOT Set   - RMSE: %.4f | MAE: %.4f",
            metrics.get("oot_rmse_t30", float("nan")),
            metrics.get("oot_mae_t30", float("nan")),
        )
        logger.info("Model saved to: %s", self.output_dir)
        logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
        logger.info("=" * 80 + "\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-features",
        type=Path,
        required=True,
        help="Path to market features CSV (Gold layer)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Path to labels CSV (Gold layer). If not provided, will be inferred from market features path.",
    )
    parser.add_argument(
        "--news-signals",
        type=Path,
        help="Path to news signals CSV (Gold layer).",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=30,
        help="Prediction horizon in minutes (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_clean/models"),
        help="Output directory for models",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sp500_prediction",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--lag-min",
        type=int,
        default=1,
        help="Minimum lag to consider during tuning (default: 1)",
    )
    parser.add_argument(
        "--lag-max",
        type=int,
        default=7,
        help="Maximum lag to consider during tuning (default: 7)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_registry_uri(args.mlflow_uri)

    pipeline = AutoregressiveOLSMLflowTrainingPipeline(
        market_features_path=args.market_features,
        labels_path=args.labels,
        news_signals_path=args.news_signals,
        prediction_horizon_minutes=args.prediction_horizon,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
