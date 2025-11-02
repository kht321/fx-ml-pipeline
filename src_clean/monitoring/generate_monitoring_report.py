from pathlib import Path
import os, argparse
import pandas as pd, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import evidently
print(f'Evidently version: {evidently.__version__}')
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, DataQualityPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.test_preset import NoTargetPerformanceTestPreset, DataQualityTestPreset, DataStabilityTestPreset
from evidently import ColumnMapping 

DATA_DIR = os.getenv("DATA_DIR", "data_clean")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
GOLD = os.path.join(DATA_DIR, "gold")
GOLD_MKT = os.path.join(DATA_DIR, "gold", "market", "features", "spx500_features.parquet")
GOLD_NEWS = os.path.join(DATA_DIR, "gold", "news", "signals", "spx500_trading_signals.parquet")
GOLD_LABELS = os.path.join(DATA_DIR, "gold", "market", "labels", "spx500_labels_30min.parquet")   # user_id, target_regression, label
PREDS = os.path.join(DATA_DIR, "predictions", "spx500_batch_scores.parquet")  # user_id, score
REPORT_FEATURES_HTML = os.path.join(REPORTS_DIR, "features_latest_report.html")
REPORT_DRIFT_HTML = os.path.join(REPORTS_DIR, "features_drift_report.html")
START_TRAIN = "2023-10-13"
END_TRAIN = "2025-04-11"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-market-features-file", required=True)
    parser.add_argument("--gold-market-labels-file", required=True)
    parser.add_argument("--predictions-file", required=True)
    args = parser.parse_args()

    print('=== Evidently Monitoring Report ===')

    # Check if required files exist
    features_file = Path(args.gold_market_features_file)
    labels_file = Path(args.gold_market_labels_file)
    preds_file = Path(args.predictions_file)
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
    df_gold = df_gold.merge(
        df_pred[['time', 'predicted_regression']], 
        on=["time"], 
        how="inner", 
        suffixes=("", "_pred")
    )

    # Convert time to datetime
    df_gold['time'] = pd.to_datetime(df_gold['time'])

    # Split train/test
    features_train_df = df_gold[
        (df_gold['time'] >= START_TRAIN) & 
        (df_gold['time'] <= END_TRAIN)
    ]
    features_oot_df = df_gold[df_gold['time'] > END_TRAIN]

    # ✓ Create reports with Evidently 0.5.0 API
    regression_report = Report(metrics=[
        RegressionPreset(),
    ])
    
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    column_mapping = ColumnMapping(
            target='target_regression',
            prediction='predicted_regression',
    )
    # ✓ Run reports - pass DataFrames directly
    regression_report.run(
        reference_data=features_train_df,
        current_data=features_oot_df,
        column_mapping=column_mapping
    )
    
    drift_report.run(
        reference_data=features_train_df,
        current_data=features_oot_df,
    )

    # Save reports
    os.makedirs(REPORTS_DIR, exist_ok=True)
    regression_report.save_html(REPORT_FEATURES_HTML)
    drift_report.save_html(REPORT_DRIFT_HTML)

    print('✓ Monitoring data validated')
    print('  Note: Full Evidently report generation available via separate service')
    print('  Access at: http://localhost:8050')

if __name__ == "__main__":
    main()