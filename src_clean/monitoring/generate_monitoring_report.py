from pathlib import Path
import os, argparse, json, argparse, pickle, sys
import pandas as pd, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import evidently
print(f'Evidently version: {evidently.__version__}')
from evidently import Report, Regression, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from evidently.tests import lte, gte, lt, gt, is_in, not_in, eq, not_eq
from evidently.metrics import RMSE, ValueDrift, DriftedColumnsCount

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data_clean")
REPORTS_DIR = os.path.join(ROOT, "reports")
GOLD = os.path.join(DATA_DIR, "gold")
GOLD_MKT = os.path.join(DATA_DIR, "gold", "market", "features", "spx500_features.parquet")
GOLD_NEWS = os.path.join(DATA_DIR, "gold", "news", "signals", "spx500_trading_signals.parquet")
GOLD_LABELS = os.path.join(DATA_DIR, "gold", "market", "labels", "spx500_labels_30min.parquet")   # user_id, target_regression, label
PREDS = os.path.join(DATA_DIR, "predictions", "spx500_batch_scores.parquet")  # user_id, score
REPORT_FEATURES_HTML = os.path.join(REPORTS_DIR, "features_latest_report.html")
REPORT_DRIFT_HTML = os.path.join(REPORTS_DIR, "features_drift_report.html")
MODELS_DIR = os.path.join(DATA_DIR, "models")
START_TRAIN = "2023-10-13"
END_TRAIN = "2025-04-11"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def load_current_model():
    model_path = os.path.join(MODELS_DIR, "production", "current_model.pkl")
    current_features_path = os.path.join(MODELS_DIR, "production", "current_features.json")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(current_features_path, "r") as f:
        current_features = json.load(f)['features']
    return model, current_features

def main():
    print("Path exposure:", os.listdir(os.getcwd()))
    print("Current Path:", os.getcwd())
    print("Docker service being used:", os.getenv("DOCKER_SERVICE_NAME", "N/A"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-market-features-file", required=True)
    parser.add_argument("--gold-market-labels-file", required=True)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--features", required=True)
    args = parser.parse_args()

    print('=== Evidently Monitoring Report ===')

    # Check if required files exist
    features_file = Path(args.gold_market_features_file)
    labels_file = Path(args.gold_market_labels_file)
    preds_file = Path(args.predictions_file)
    model_selected_features = Path(args.features)

    if not features_file.exists() or not labels_file.exists() or not preds_file.exists():
        print('⚠ Features or labels or predictions file not found, skipping monitoring report')
        print('  (This is expected on first run)')
        exit(0)

    # ✓ Read docker is pushing CSV files
    if features_file.suffix == '.parquet':
        df_gold_mkt = pd.read_parquet(features_file)
    else:
        df_gold_mkt = pd.read_csv(features_file)

    if labels_file.suffix == '.parquet':
        df_gold_labels = pd.read_parquet(labels_file)
    else:
        df_gold_labels = pd.read_csv(labels_file)
    if preds_file.suffix == '.parquet':
        df_pred = pd.read_parquet(preds_file)
    else:
        df_pred = pd.read_csv(preds_file)
    print(f'✓ Loaded {len(df_gold_mkt):,} rows for monitoring')

    # Simple data profile
    print(f'✓ Features: {len(df_gold_mkt.columns)} columns')
    print(f'✓ Date range: {df_gold_mkt.time.min() if "time" in df_gold_mkt.columns else "N/A"} to {df_gold_mkt.time.max() if "time" in df_gold_mkt.columns else "N/A"}')
    print(f'✓ Missing values: {df_gold_mkt.isnull().sum().sum()} total')

    # Merge data
    df_gold = df_gold_mkt.merge(
        df_gold_labels[['time', 'target_regression']], 
        on=["time"], 
        how="inner", 
        suffixes=("_mkt", "_labels")
    )

    # Convert time to datetime
    df_gold['time'] = pd.to_datetime(df_gold['time'])
    model, current_features = load_current_model()
    predicted_regression = model.predict(df_gold[current_features])
    df_gold['predicted_regression'] = predicted_regression

    numerical_features = [d for d in df_gold.columns if df_gold[d].dtype in [np.float64, np.int64] and d not in ["time", "predicted_regression", "target_regression"]]
    categorical_features = [d for d in df_gold.columns if df_gold[d].dtype == object and d not in ["time", "predicted_regression", "target_regression"]]
    numerical_features = [f for f in numerical_features if f in current_features]
    categorical_features = [f for f in categorical_features if f in current_features]

    # Create dataset
    data_definition = DataDefinition(
        regression=[Regression(target="target_regression", prediction="predicted_regression")],
        numerical_columns=numerical_features,
        categorical_columns=categorical_features
    )

    features_train_df = df_gold[(df_gold['time'] >= START_TRAIN) & (df_gold['time'] <= END_TRAIN)][numerical_features + categorical_features + ['time', 'target_regression', 'predicted_regression']]
    features_oot_df = df_gold[df_gold['time'] > END_TRAIN][numerical_features + categorical_features + ['time', 'target_regression', 'predicted_regression']]
    reference_dataset = Dataset.from_pandas(features_train_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(features_oot_df, data_definition=data_definition)

    regression_preset = Report(metrics=[RegressionPreset()])
    regression_snapshot_with_reference = regression_preset.run(current_data=current_dataset, reference_data=current_dataset)

    # Drift report
    value_drift_columns = [ValueDrift(column=col, method="psi", threshold=0.05) for col in numerical_features]
    drift_report = Report([
        DriftedColumnsCount(
            cat_stattest="psi", num_stattest="wasserstein",
            per_column_method={"target_regression": "psi", "predicted_regression": "psi"}, drift_share=0.8
        )
    ] + value_drift_columns, include_tests=False)

    drift_snapshot = drift_report.run(current_data=current_dataset, reference_data=reference_dataset)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    regression_snapshot_with_reference.save_html(REPORT_FEATURES_HTML)
    drift_snapshot.save_html(REPORT_DRIFT_HTML)
    
    print(f'✓ Saved monitoring reports to {REPORT_FEATURES_HTML} and {REPORT_DRIFT_HTML}')

    return REPORT_FEATURES_HTML, REPORT_DRIFT_HTML

if __name__ == "__main__":
    main()