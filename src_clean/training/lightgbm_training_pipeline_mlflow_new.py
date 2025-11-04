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
from __future__ import annotations

import mlflow.lightgbm
import lightgbm as lgb
import optuna
import numpy as np
from typing import Dict, Tuple, Optional, List

from .xgboost_training_pipeline_mlflow import XGBoostMLflowTrainingPipeline
from sklearn.metrics import roc_auc_score

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
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
import mlflow.lightgbm
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




class LightGBMMLflowTrainingPipeline(XGBoostMLflowTrainingPipeline):
    """MLflow-enabled LightGBM pipeline reusing the FX pipeline stack."""

    def __init__(
        self,
        *args,
        enable_tuning: bool = False,
        stage1_n_trials: int = 5,
        stage2_n_trials: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            enable_tuning=enable_tuning,
            stage1_n_trials=stage1_n_trials,
            stage2_n_trials=stage2_n_trials,
            **kwargs,
        )
        self.lgb_params = {
            "classification": {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.05,
                "num_leaves": 64,
                "max_depth": -1,
                "n_estimators": 600,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
            },
            "regression": {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 128,
                "max_depth": -1,
                "n_estimators": 800,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            },
        }

        self.model_name = "lightgbm"

    def train_model(
        self,
        X,
        y,
        time_index,
        feature_names: List[str],
    ) -> Tuple[lgb.LGBMModel, Dict]:
        logger.info("Training LGBM model with MLflow tracking...")

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

        with mlflow.start_run(
            run_name=f"{self.experiment_name}_LGBM_{self.prediction_horizon_minutes}min"
        ):
            mlflow.log_param("task_type", self.task)
            mlflow.log_param("prediction_horizon_minutes", self.prediction_horizon)
            mlflow.log_param("n_total_samples", len(X))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_oot_samples", len(X_oot))
            mlflow.log_param("n_total_features", len(feature_names))
            mlflow.log_param("news_available", int("news_available" in X.columns))

            test_size = int(0.15 * X_train_val.shape[0])
            tscv = TimeSeriesSplit(n_splits=3, test_size=test_size)

            if self.enable_tuning:
                model = self._tune_hyperparameters(X_train_val, y_train_val, tscv)
                selected_feature_names = getattr(self, "selected_feature_names", feature_names)
                self.selected_feature_names = selected_feature_names
                logger.info(f"Selected {len(selected_feature_names)} features after tuning.")
                logger.info(f"Selected features are: {selected_feature_names}")
            else:
                ModelClass = lgb.LGBMClassifier if self.task == "classification" else lgb.LGBMRegressor
                params = self.lgb_params[self.task].copy()
                model = ModelClass(**params)
                selected_feature_names = feature_names
                mlflow.log_params(params)
                mlflow.log_param("selected_feature_group", "all")
                mlflow.log_param("n_selected_features", len(selected_feature_names))

            cv_scores = []
            for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_val), 1):
                X_tr = X_train_val.iloc[tr_idx][selected_feature_names]
                y_tr = y_train_val.iloc[tr_idx]
                X_va = X_train_val.iloc[va_idx][selected_feature_names]
                y_va = y_train_val.iloc[va_idx]

                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="auc" if self.task == "classification" else "rmse",
                    callbacks=[lgb.early_stopping(100)],
                )

                if self.task == "classification":
                    proba = model.predict_proba(X_va)[:, 1]
                    score = roc_auc_score(y_va, proba)
                else:
                    preds = model.predict(X_va)
                    score = np.sqrt(np.mean((preds - y_va) ** 2))

                cv_scores.append(float(score))
                mlflow.log_metric(f"cv_fold_{fold}_score", float(score))

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            mlflow.log_metric("cv_mean", float(cv_mean))
            mlflow.log_metric("cv_std", float(cv_std))

            # Train final model on train+val
            logger.info("Training final model on train+val...")
            model.fit(X_train_val[selected_feature_names], y_train_val)

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
            mlflow.lightgbm.log_model(model, "model")

            # Log feature names
            mlflow.log_dict({"features": selected_feature_names}, "features.json")

            # Save feature importance plot
            self._plot_feature_importance_mlflow(model, selected_feature_names)

            logger.info(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
            logger.info("="*80 + "\n")
            return model, metrics

    def _tune_hyperparameters(self, X, y, tscv):
        ModelClass = lgb.LGBMClassifier if self.task == "classification" else lgb.LGBMRegressor
        base_params = self.lgb_params[self.task].copy()
        direction = "maximize" if self.task == "classification" else "minimize"
        metric_name = "auc" if self.task == "classification" else "rmse"
        X_df = X.copy()
        y_series = y.copy()

        stage1_trials = getattr(self, "stage1_n_trials", 5)
        stage2_trials = getattr(self, "stage2_n_trials", 5)

        # Ensure hyperparameter artifact directory exists
        hyper_dir = getattr(self, "hyperparameter_dir", self.output_dir / "hyperparameters" / self.experiment_name)
        hyper_dir.mkdir(parents=True, exist_ok=True)

        def _suggest_params(trial: optuna.Trial) -> Dict[str, float]:
            lr = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
            num_leaves = trial.suggest_int("num_leaves", 16, 512, log=True)
            max_depth = trial.suggest_int("max_depth", -1, 16)
            min_child_samples = trial.suggest_int("min_child_samples", 20, 200)
            lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True)
            lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 20.0, log=True)
            feature_frac = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            bagging_frac = trial.suggest_float("subsample", 0.5, 1.0)
            bagging_freq = trial.suggest_int("subsample_freq", 1, 10)
            min_split_gain = trial.suggest_float("min_split_gain", 0.0, 5.0)
            max_bin = trial.suggest_int("max_bin", 64, 512)

            params = {
                "verbosity": -1,
                "learning_rate": lr,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "min_child_samples": min_child_samples,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "colsample_bytree": feature_frac,
                "subsample": bagging_frac,
                "subsample_freq": bagging_freq,
                "min_split_gain": min_split_gain,
                "max_bin": max_bin,
            }

            if self.task == "classification":
                params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])

            params["n_estimators"] = trial.suggest_int("n_estimators", 400, 5000)
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

                model.fit(X_train, y_train)

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
        stage1_model.fit(X_df, y_series)

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
        logger.info("Stage 2 complete: best %s=%.5f with params=%s and feature_group=%s",
                    metric_name, stage2_study.best_value, best_stage2_params, best_group)
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
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
        type=str,
        default="False",
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--stage1-n-trials",
        type=int,
        default=5,
        help="Number of trials for stage 1"
    )
    parser.add_argument(
        "--stage2-n-trials",
        type=int,
        default=5,
        help="Number of trials for stage 2"
    )
    parser.add_argument(
        "--accelerate-dataset",
        type=str,
        default="False",
        help="Use a smaller dataset for faster testing"
    )

    args = parser.parse_args()
    args.accelerate_dataset = args.accelerate_dataset.lower() == "true"
    args.enable_tuning = args.enable_tuning.lower() == "true"

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
        enable_tuning=args.enable_tuning,
        accelerate_dataset=args.accelerate_dataset,
        stage1_n_trials=args.stage1_n_trials,
        stage2_n_trials=args.stage2_n_trials,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
