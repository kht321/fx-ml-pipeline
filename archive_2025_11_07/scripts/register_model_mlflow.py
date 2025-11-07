#!/usr/bin/env python3
"""
Register trained XGBoost model in MLflow Model Registry.

This script registers the FinBERT-enhanced XGBoost model to MLflow,
enabling version control, deployment tracking, and model lifecycle management.
"""

import mlflow
import mlflow.sklearn
import pickle
import json
from pathlib import Path
import sys
from datetime import datetime

# Add src_clean to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src_clean"))


def register_model_to_mlflow(
    model_path: str,
    metrics_path: str,
    features_path: str,
    model_name: str = "xgboost_finbert_classifier",
    description: str = None,
):
    """
    Register a trained model to MLflow Model Registry.

    Args:
        model_path: Path to pickle file
        metrics_path: Path to metrics JSON
        features_path: Path to features JSON
        model_name: Name for the registered model
        description: Model description
    """
    print("=" * 80)
    print("MLflow Model Registration")
    print("=" * 80)

    # Load model
    print(f"\n1. Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"   ✓ Model loaded: {type(model).__name__}")

    # Load metrics
    print(f"\n2. Loading metrics from: {metrics_path}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    print(f"   ✓ Metrics loaded:")
    print(f"     - AUC: {metrics['auc']:.4f}")
    print(f"     - Accuracy: {metrics['accuracy']:.4f}")
    print(f"     - CV Mean: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

    # Load features
    print(f"\n3. Loading features from: {features_path}")
    with open(features_path, 'r') as f:
        features = json.load(f)
    print(f"   ✓ Features loaded: {len(features)} features")

    # Set MLflow tracking URI (local directory)
    mlflow_dir = Path(__file__).parent.parent / "mlruns"
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
    print(f"\n4. MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Create experiment
    experiment_name = "xgboost_finbert_pipeline"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"   ✓ Created new experiment: {experiment_name}")
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"   ✓ Using existing experiment: {experiment_name}")

    # Start MLflow run
    print(f"\n5. Starting MLflow run...")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"   ✓ Run ID: {run_id}")

        # Log parameters
        print(f"\n6. Logging parameters...")
        params = {
            "model_type": "XGBoost",
            "task": "classification",
            "prediction_horizon": "30min",
            "features_count": len(features),
            "has_finbert": True,
            "training_date": datetime.now().strftime("%Y-%m-%d"),
        }
        mlflow.log_params(params)
        print(f"   ✓ Logged {len(params)} parameters")

        # Log metrics
        print(f"\n7. Logging metrics...")
        mlflow_metrics = {
            "auc": metrics["auc"],
            "accuracy": metrics["accuracy"],
            "cv_mean": metrics["cv_mean"],
            "cv_std": metrics["cv_std"],
            "f1_macro": metrics["classification_report"]["macro avg"]["f1-score"],
            "precision_macro": metrics["classification_report"]["macro avg"]["precision"],
            "recall_macro": metrics["classification_report"]["macro avg"]["recall"],
        }
        mlflow.log_metrics(mlflow_metrics)
        print(f"   ✓ Logged {len(mlflow_metrics)} metrics")

        # Log artifacts
        print(f"\n8. Logging artifacts...")
        model_dir = Path(model_path).parent
        mlflow.log_artifact(metrics_path, "metrics")
        mlflow.log_artifact(features_path, "config")

        feature_importance_csv = model_dir / f"{Path(model_path).stem}_feature_importance.csv"
        if feature_importance_csv.exists():
            mlflow.log_artifact(str(feature_importance_csv), "feature_importance")
            print(f"   ✓ Logged feature importance CSV")

        feature_importance_png = model_dir / f"{Path(model_path).stem}_feature_importance.png"
        if feature_importance_png.exists():
            mlflow.log_artifact(str(feature_importance_png), "visualizations")
            print(f"   ✓ Logged feature importance plot")

        # Log model
        print(f"\n9. Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        print(f"   ✓ Model logged and registered as: {model_name}")

        # Add model description
        if description:
            client = mlflow.tracking.MlflowClient()
            client.update_registered_model(
                name=model_name,
                description=description
            )
            print(f"   ✓ Added model description")

        print(f"\n{'=' * 80}")
        print(f"✓ Model Registration Complete!")
        print(f"{'=' * 80}")
        print(f"\nModel Details:")
        print(f"  - Name: {model_name}")
        print(f"  - Run ID: {run_id}")
        print(f"  - AUC: {metrics['auc']:.4f}")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Features: {len(features)}")
        print(f"\nView in MLflow UI:")
        print(f"  mlflow ui --backend-store-uri {mlflow_dir.absolute()}")
        print(f"  http://localhost:5000")
        print(f"\n{'=' * 80}")

        return run_id, model_name


if __name__ == "__main__":
    # Default paths (5-year model)
    model_dir = Path(__file__).parent.parent / "data_clean_5year" / "models"

    # Find the latest model
    pkl_files = list(model_dir.glob("*.pkl"))
    if not pkl_files:
        print("Error: No .pkl files found in data_clean_5year/models/")
        sys.exit(1)

    # Use the most recent model
    latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime)
    model_stem = latest_pkl.stem

    model_path = str(latest_pkl)
    metrics_path = str(model_dir / f"{model_stem}_metrics.json")
    features_path = str(model_dir / f"{model_stem}_features.json")

    # Verify all files exist
    for path in [model_path, metrics_path, features_path]:
        if not Path(path).exists():
            print(f"Error: Required file not found: {path}")
            sys.exit(1)

    # Model description
    description = """
    XGBoost Classifier with FinBERT News Sentiment Features

    This model predicts 30-minute ahead price movements (Up/Down) for SPX500
    using market technical indicators, microstructure features, volatility metrics,
    and FinBERT-derived news sentiment signals.

    Key Features:
    - Market Technical: SMA, EMA, RSI, MACD, Bollinger Bands
    - Market Microstructure: Bid-ask spreads, volume, order flow
    - Market Volatility: ATR, realized volatility, Garman-Klass
    - News Sentiment: FinBERT financial sentiment (60-min aggregation)

    Training Data:
    - SPX500 5-year historical data (2020-2025)
    - 12,950 news articles processed
    - 8,769 trading signals generated
    - 1.7M training samples

    Performance:
    - AUC: 0.636
    - Accuracy: 58.92%
    - Cross-validation: 50.68% ± 0.66%
    """

    # Register model
    register_model_to_mlflow(
        model_path=model_path,
        metrics_path=metrics_path,
        features_path=features_path,
        model_name="xgboost_finbert_classifier",
        description=description.strip(),
    )
