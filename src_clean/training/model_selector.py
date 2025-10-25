"""
Model Selector - Select Best Model from MLflow Experiments

Repository Location: fx-ml-pipeline/src_clean/training/model_selector.py

Purpose:
    Selects the best performing model from MLflow experiments based on:
    1. OOT (Out-of-Time) performance
    2. Test performance
    3. No severe overfitting
    4. Promotes best model to production

Selection Criteria:
    - OOT AUC >= 0.55 (minimum threshold)
    - Test AUC >= 0.55 (minimum threshold)
    - Overfitting check: (train_auc - test_auc) < 0.10
    - Selects model with highest OOT AUC among qualifying models

Usage:
    # Automatic selection (uses default experiment)
    python src_clean/training/model_selector.py

    # Specify experiment
    python src_clean/training/model_selector.py \\
        --experiment-name sp500_prediction \\
        --metric oot_auc \\
        --min-threshold 0.55

    # Promote to production immediately
    python src_clean/training/model_selector.py --promote

Output:
    - Displays ranked models
    - Optionally promotes best to data_clean/models/production/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
import shutil
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelSelector:
    """Select best model from MLflow experiments based on performance."""

    def __init__(
        self,
        experiment_name: str = "sp500_prediction",
        mlflow_uri: str = "mlruns",
        production_dir: Path = Path("data_clean/models/production")
    ):
        """
        Initialize model selector.

        Parameters
        ----------
        experiment_name : str
            MLflow experiment name
        mlflow_uri : str
            MLflow tracking URI
        production_dir : Path
            Directory for production models
        """
        self.experiment_name = experiment_name
        self.production_dir = production_dir
        self.production_dir.mkdir(parents=True, exist_ok=True)

        # Set up MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient()

    def get_all_runs(self) -> List[Dict]:
        """Get all runs from experiment."""
        logger.info(f"Fetching runs from experiment: {self.experiment_name}")

        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment '{self.experiment_name}' not found")

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=100
            )

            run_data = []
            for run in runs:
                run_info = {
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                    'status': run.info.status,
                    'model_type': run.data.params.get('model_type', 'Unknown'),
                    'task_type': run.data.params.get('task_type', 'Unknown'),
                    'train_auc': run.data.metrics.get('train_auc', 0),
                    'val_auc': run.data.metrics.get('val_auc', 0),
                    'test_auc': run.data.metrics.get('test_auc', 0),
                    'oot_auc': run.data.metrics.get('oot_auc', 0),
                    'train_accuracy': run.data.metrics.get('train_accuracy', 0),
                    'test_accuracy': run.data.metrics.get('test_accuracy', 0),
                    'oot_accuracy': run.data.metrics.get('oot_accuracy', 0),
                    'n_features': run.data.params.get('n_features', 0)
                }
                run_data.append(run_info)

            logger.info(f"Found {len(run_data)} runs")
            return run_data

        except Exception as e:
            logger.error(f"Error fetching runs: {e}")
            return []

    def select_best_model(
        self,
        metric: str = "oot_auc",
        min_threshold: float = 0.55,
        max_overfitting: float = 0.10
    ) -> Optional[Dict]:
        """
        Select best model based on criteria.

        Selection Logic:
        ----------------
        1. Filter runs with metric >= min_threshold
        2. Filter runs with test metric >= min_threshold
        3. Check overfitting: (train - test) < max_overfitting
        4. Sort by OOT metric (highest first)
        5. Return best qualifying run

        Parameters
        ----------
        metric : str
            Primary metric for selection (default: oot_auc)
        min_threshold : float
            Minimum acceptable value for metric
        max_overfitting : float
            Maximum acceptable overfitting (train - test)

        Returns
        -------
        dict or None
            Best run metadata, or None if no qualifying runs
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL SELECTION")
        logger.info("="*80)
        logger.info(f"Primary metric: {metric}")
        logger.info(f"Minimum threshold: {min_threshold}")
        logger.info(f"Max overfitting: {max_overfitting}")

        runs = self.get_all_runs()

        if not runs:
            logger.error("No runs found in experiment")
            return None

        # Convert to DataFrame for easier filtering
        runs_df = pd.DataFrame(runs)

        # Apply filters
        logger.info("\nApplying selection criteria...")

        # Filter 1: Primary metric >= threshold
        qualified = runs_df[runs_df[metric] >= min_threshold]
        logger.info(f"  After {metric} >= {min_threshold}: {len(qualified)} runs")

        if len(qualified) == 0:
            logger.warning(f"No runs meet minimum {metric} threshold of {min_threshold}")
            logger.info("\nTop 5 runs by OOT AUC:")
            logger.info(runs_df.nlargest(5, 'oot_auc')[
                ['run_id', 'model_type', 'oot_auc', 'test_auc', 'train_auc']
            ].to_string())
            return None

        # Filter 2: Test metric >= threshold
        test_metric = metric.replace('oot_', 'test_')
        qualified = qualified[qualified[test_metric] >= min_threshold]
        logger.info(f"  After {test_metric} >= {min_threshold}: {len(qualified)} runs")

        if len(qualified) == 0:
            logger.warning(f"No runs meet minimum {test_metric} threshold of {min_threshold}")
            return None

        # Filter 3: Check overfitting
        train_metric = metric.replace('oot_', 'train_')
        qualified['overfitting'] = qualified[train_metric] - qualified[test_metric]
        qualified = qualified[qualified['overfitting'] < max_overfitting]
        logger.info(f"  After overfitting < {max_overfitting}: {len(qualified)} runs")

        if len(qualified) == 0:
            logger.warning("No runs meet overfitting criteria")
            return None

        # Sort by primary metric
        qualified = qualified.sort_values(by=metric, ascending=False)

        # Get best run
        best_run = qualified.iloc[0].to_dict()

        logger.info("\n" + "="*80)
        logger.info("BEST MODEL SELECTED")
        logger.info("="*80)
        logger.info(f"Run ID: {best_run['run_id']}")
        logger.info(f"Model Type: {best_run['model_type']}")
        logger.info(f"Task Type: {best_run['task_type']}")
        logger.info(f"Trained: {best_run['start_time']}")
        logger.info(f"\nPerformance:")
        logger.info(f"  Train AUC: {best_run['train_auc']:.4f}")
        logger.info(f"  Val AUC:   {best_run['val_auc']:.4f}")
        logger.info(f"  Test AUC:  {best_run['test_auc']:.4f}")
        logger.info(f"  OOT AUC:   {best_run['oot_auc']:.4f}")
        logger.info(f"  Overfitting: {best_run['overfitting']:.4f}")
        logger.info("="*80 + "\n")

        # Display top 5 for comparison
        logger.info("Top 5 Qualifying Models:")
        display_cols = ['run_id', 'model_type', 'oot_auc', 'test_auc', 'overfitting']
        logger.info(qualified[display_cols].head(5).to_string(index=False))

        return best_run

    def promote_to_production(
        self,
        run_id: str,
        create_backup: bool = True
    ) -> Dict:
        """
        Promote model to production.

        Steps:
        ------
        1. Backup existing production model (if exists)
        2. Copy model artifacts from MLflow
        3. Save model metadata
        4. Update production symlink

        Parameters
        ----------
        run_id : str
            MLflow run ID to promote
        create_backup : bool
            Whether to backup existing production model

        Returns
        -------
        dict
            Promotion metadata
        """
        logger.info("\n" + "="*80)
        logger.info("PROMOTING MODEL TO PRODUCTION")
        logger.info("="*80)

        # Backup existing production model
        if create_backup and (self.production_dir / "best_model.pkl").exists():
            backup_dir = self.production_dir.parent / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Backing up existing production model to: {backup_dir}")

            for file in self.production_dir.glob("*"):
                if file.is_file():
                    shutil.copy2(file, backup_dir / file.name)

        # Get run details
        run = self.client.get_run(run_id)

        # Copy model artifacts
        logger.info(f"Copying model artifacts from run {run_id}...")

        # Download model from MLflow
        model_uri = f"runs:/{run_id}/model"

        try:
            # For models logged with mlflow.xgboost.log_model
            model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=str(self.production_dir)
            )
            logger.info(f"  Model downloaded to: {model_path}")
        except Exception as e:
            logger.error(f"  Failed to download model: {e}")
            logger.info("  Trying alternative method...")

            # Alternative: copy from mlruns directory
            artifact_path = run.info.artifact_uri.replace("file://", "")
            model_src = Path(artifact_path) / "model"

            if model_src.exists():
                model_dst = self.production_dir / "model"
                shutil.copytree(model_src, model_dst, dirs_exist_ok=True)
                logger.info(f"  Model copied to: {model_dst}")
            else:
                raise FileNotFoundError(f"Model artifacts not found at {model_src}")

        # Create production metadata
        metadata = {
            "run_id": run_id,
            "run_name": run.data.tags.get('mlflow.runName', 'Unknown'),
            "model_type": run.data.params.get('model_type', 'Unknown'),
            "task_type": run.data.params.get('task_type', 'Unknown'),
            "train_auc": run.data.metrics.get('train_auc', 0),
            "val_auc": run.data.metrics.get('val_auc', 0),
            "test_auc": run.data.metrics.get('test_auc', 0),
            "oot_auc": run.data.metrics.get('oot_auc', 0),
            "trained_at": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
            "promoted_at": datetime.now().isoformat(),
            "model_version": run_id[:8],
            "n_features": run.data.params.get('n_features', 0)
        }

        # Save metadata
        metadata_path = self.production_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Metadata saved: {metadata_path}")

        # Copy feature config if available
        try:
            features_uri = f"runs:/{run_id}/features.json"
            features_path = mlflow.artifacts.download_artifacts(
                artifact_uri=features_uri,
                dst_path=str(self.production_dir)
            )
            logger.info(f"  Features config saved: {features_path}")
        except:
            logger.warning("  Features config not available")

        logger.info("\n✓ Model successfully promoted to production!")
        logger.info(f"  Location: {self.production_dir}")
        logger.info(f"  Model version: {metadata['model_version']}")
        logger.info(f"  OOT AUC: {metadata['oot_auc']:.4f}")
        logger.info("="*80 + "\n")

        return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sp500_prediction",
        help="MLflow experiment name"
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
        default=0.55,
        help="Minimum threshold for metric (default: 0.55)"
    )
    parser.add_argument(
        "--max-overfitting",
        type=float,
        default=0.10,
        help="Maximum acceptable overfitting (default: 0.10)"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote best model to production"
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path("data_clean/models/production"),
        help="Production model directory"
    )

    args = parser.parse_args()

    # Initialize selector
    selector = ModelSelector(
        experiment_name=args.experiment_name,
        mlflow_uri=args.mlflow_uri,
        production_dir=args.production_dir
    )

    # Select best model
    best_model = selector.select_best_model(
        metric=args.metric,
        min_threshold=args.min_threshold,
        max_overfitting=args.max_overfitting
    )

    if best_model is None:
        logger.error("No qualifying model found. Cannot promote to production.")
        return 1

    # Promote if requested
    if args.promote:
        selector.promote_to_production(best_model['run_id'])
    else:
        logger.info("\nTo promote this model to production, run:")
        logger.info(f"  python src_clean/training/model_selector.py --promote")

    return 0


if __name__ == "__main__":
    exit(main())
