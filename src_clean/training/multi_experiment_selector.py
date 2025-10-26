"""
Multi-Experiment Model Selector - Compare models across multiple MLflow experiments

Compares models from different experiments (XGBoost, LightGBM, Transformer) and selects
the best performing model based on OOT (Out-of-Time) performance.

Usage:
    python src_clean/training/multi_experiment_selector.py \\
        --experiments sp500_5year_prediction sp500_5year_prediction_lightgbm sp500_5year_prediction_transformer \\
        --metric oot_auc \\
        --min-threshold 0.50

Repository Location: fx-ml-pipeline/src_clean/training/multi_experiment_selector.py
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiExperimentSelector:
    """Select best model across multiple MLflow experiments."""

    def __init__(self, mlflow_uri: str = "mlruns"):
        """Initialize selector."""
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient()

    def get_runs_from_experiments(self, experiment_names: List[str]) -> pd.DataFrame:
        """Get all runs from multiple experiments."""
        all_runs = []

        for exp_name in experiment_names:
            try:
                experiment = self.client.get_experiment_by_name(exp_name)
                if experiment is None:
                    logger.warning(f"Experiment '{exp_name}' not found, skipping")
                    continue

                runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=100
                )

                for run in runs:
                    run_data = {
                        'experiment_name': exp_name,
                        'run_id': run.info.run_id,
                        'model_type': run.data.params.get('model_type', 'xgboost'),  # Default to xgboost for backward compat
                        'task_type': run.data.params.get('task_type', 'Unknown'),
                        'train_auc': run.data.metrics.get('train_auc', 0),
                        'val_auc': run.data.metrics.get('val_auc', 0),
                        'test_auc': run.data.metrics.get('test_auc', 0),
                        'oot_auc': run.data.metrics.get('oot_auc', 0),
                        'train_accuracy': run.data.metrics.get('train_accuracy', 0),
                        'test_accuracy': run.data.metrics.get('test_accuracy', 0),
                        'oot_accuracy': run.data.metrics.get('oot_accuracy', 0),
                        'n_features': run.data.params.get('n_features', 0),
                        'status': run.info.status
                    }
                    all_runs.append(run_data)

                logger.info(f"Loaded {len(runs)} runs from {exp_name}")

            except Exception as e:
                logger.error(f"Error loading experiment {exp_name}: {e}")
                continue

        if not all_runs:
            logger.error("No runs found in any experiment")
            return pd.DataFrame()

        df = pd.DataFrame(all_runs)
        logger.info(f"\nTotal runs across all experiments: {len(df)}")
        return df

    def compare_models(
        self,
        experiment_names: List[str],
        metric: str = "oot_auc",
        min_threshold: float = 0.50,
        max_overfitting: float = 0.15
    ) -> Optional[Dict]:
        """
        Compare models across experiments and select best.

        Parameters
        ----------
        experiment_names : List[str]
            List of experiment names to compare
        metric : str
            Primary metric for selection (default: oot_auc)
        min_threshold : float
            Minimum acceptable value for metric
        max_overfitting : float
            Maximum acceptable overfitting (train - test)

        Returns
        -------
        dict or None
            Best model metadata
        """
        logger.info("\n" + "="*80)
        logger.info("MULTI-EXPERIMENT MODEL COMPARISON")
        logger.info("="*80)
        logger.info(f"Experiments: {', '.join(experiment_names)}")
        logger.info(f"Primary metric: {metric}")
        logger.info(f"Minimum threshold: {min_threshold}")
        logger.info(f"Max overfitting: {max_overfitting}")

        # Get all runs
        df = self.get_runs_from_experiments(experiment_names)

        if df.empty:
            logger.error("No runs to compare")
            return None

        # Apply filters
        logger.info("\nApplying selection criteria...")

        # Filter 1: Completed runs only
        df = df[df['status'] == 'FINISHED']
        logger.info(f"  After filtering completed runs: {len(df)} runs")

        # Filter 2: Primary metric >= threshold
        qualified = df[df[metric] >= min_threshold]
        logger.info(f"  After {metric} >= {min_threshold}: {len(qualified)} runs")

        if len(qualified) == 0:
            logger.warning(f"\nNo runs meet minimum {metric} threshold of {min_threshold}")
            logger.info("\n=== ALL MODELS (Top 10 by OOT AUC) ===")
            top_models = df.nlargest(10, 'oot_auc')
            display_cols = ['model_type', 'experiment_name', 'oot_auc', 'test_auc', 'train_auc', 'oot_accuracy']
            logger.info("\n" + top_models[display_cols].to_string(index=False))
            return None

        # Filter 3: Test metric >= threshold
        test_metric = metric.replace('oot_', 'test_')
        qualified = qualified[qualified[test_metric] >= min_threshold]
        logger.info(f"  After {test_metric} >= {min_threshold}: {len(qualified)} runs")

        if len(qualified) == 0:
            logger.warning(f"No runs meet minimum {test_metric} threshold")
            return None

        # Filter 4: Check overfitting
        train_metric = metric.replace('oot_', 'train_')
        qualified['overfitting'] = qualified[train_metric] - qualified[test_metric]
        qualified = qualified[qualified['overfitting'] < max_overfitting]
        logger.info(f"  After overfitting < {max_overfitting}: {len(qualified)} runs")

        if len(qualified) == 0:
            logger.warning("No runs meet overfitting criteria")
            logger.info("\n=== ALL MODELS (Top 10 by OOT AUC) ===")
            top_models = df.nlargest(10, 'oot_auc')
            top_models['overfitting'] = top_models[train_metric] - top_models[test_metric]
            display_cols = ['model_type', 'experiment_name', 'oot_auc', 'test_auc', 'overfitting']
            logger.info("\n" + top_models[display_cols].to_string(index=False))
            return None

        # Sort by primary metric
        qualified = qualified.sort_values(by=metric, ascending=False)

        # Get best model
        best = qualified.iloc[0]

        logger.info("\n" + "="*80)
        logger.info("BEST MODEL SELECTED")
        logger.info("="*80)
        logger.info(f"Model Type: {best['model_type']}")
        logger.info(f"Experiment: {best['experiment_name']}")
        logger.info(f"Run ID: {best['run_id']}")
        logger.info(f"\nPerformance:")
        logger.info(f"  Train AUC:    {best['train_auc']:.4f}")
        logger.info(f"  Val AUC:      {best['val_auc']:.4f}")
        logger.info(f"  Test AUC:     {best['test_auc']:.4f}")
        logger.info(f"  OOT AUC:      {best['oot_auc']:.4f}")
        logger.info(f"  OOT Accuracy: {best['oot_accuracy']:.4f}")
        logger.info(f"  Overfitting:  {best['overfitting']:.4f}")
        logger.info("="*80 + "\n")

        # Display comparison of all qualified models
        logger.info("=== TOP 5 QUALIFYING MODELS ===")
        display_cols = ['model_type', 'experiment_name', 'oot_auc', 'test_auc', 'overfitting']
        logger.info("\n" + qualified[display_cols].head(5).to_string(index=False))

        # Display comparison by model type
        logger.info("\n=== BEST MODEL PER TYPE ===")
        best_per_type = qualified.groupby('model_type').first().reset_index()
        logger.info("\n" + best_per_type[display_cols].to_string(index=False))

        return best.to_dict()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "sp500_5year_prediction",
            "sp500_5year_prediction_lightgbm",
            "sp500_5year_prediction_transformer"
        ],
        help="List of experiment names to compare"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="oot_auc",
        help="Primary metric for selection (default: oot_auc)"
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.50,
        help="Minimum threshold for metric (default: 0.50)"
    )
    parser.add_argument(
        "--max-overfitting",
        type=float,
        default=0.15,
        help="Maximum acceptable overfitting (default: 0.15)"
    )

    args = parser.parse_args()

    # Initialize selector
    selector = MultiExperimentSelector(mlflow_uri=args.mlflow_uri)

    # Compare and select
    best_model = selector.compare_models(
        experiment_names=args.experiments,
        metric=args.metric,
        min_threshold=args.min_threshold,
        max_overfitting=args.max_overfitting
    )

    if best_model is None:
        logger.warning("\nNo qualifying model found based on selection criteria.")
        logger.info("Consider:")
        logger.info("  1. Lowering --min-threshold")
        logger.info("  2. Increasing --max-overfitting")
        logger.info("  3. Training more models")
        return 1

    logger.info("\nTo view this run in MLflow UI:")
    logger.info(f"  mlflow ui --backend-store-uri {args.mlflow_uri}")
    logger.info(f"  Then navigate to run: {best_model['run_id']}")

    return 0


if __name__ == "__main__":
    exit(main())
