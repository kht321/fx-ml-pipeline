import os
import pandas as pd, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from evidently import Report, Regression, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset
from evidently.tests import lte, gte, lt, gt, is_in, not_in, eq, not_eq
from evidently.metrics import RMSE, ValueDrift, DriftedColumnsCount

# USE CONTAINER PATHS (compose mounts these)
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
END_TRAIN = "2025-08-30"

app = FastAPI(title="Evidently 0.6.7 Monitor")

# Ensure the reports dir exists before mounting as static
os.makedirs(REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

def build_report() -> str:
    if not (os.path.exists(GOLD) and os.path.exists(PREDS)):
        raise FileNotFoundError("Need gold/features.csv and predictions/batch_scores.csv")

    df_gold_mkt = pd.read_parquet(GOLD_MKT)
    df_gold_labels = pd.read_parquet(GOLD_LABELS)
    # df_gold_news = pd.read_parquet(GOLD_NEWS)
    df_pred = pd.read_parquet(PREDS)
    df_gold = df_gold_mkt.merge(df_gold_labels[['time', 'target_regression']], on=["time"], how="inner", suffixes=("_mkt", "_labels"))
    df_gold = df_gold.merge(df_pred[['time', 'predicted_regression']], on=["time"], how="inner", suffixes=("", "_pred"))

    numerical_features = [d for d in df_gold.columns if df_gold[d].dtype in [np.float64, np.int64] and d not in ["time", "predicted_regression", "target_regression"]]
    categorical_features = [d for d in df_gold.columns if df_gold[d].dtype == object and d not in ["time", "predicted_regression", "target_regression"]]

    # Create dataset
    data_definition = DataDefinition(
        regression=[Regression(target="target_regression", prediction="predicted_regression")],
        numerical_columns=numerical_features,
        categorical_columns=categorical_features
    )

    features_train_df = df_gold[(df_gold['time'] >= START_TRAIN) & (df_gold['time'] <= END_TRAIN)]
    features_oot_df = df_gold[df_gold['time'] > END_TRAIN]
    reference_dataset = Dataset.from_pandas(features_train_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(features_oot_df, data_definition=data_definition)

    regression_preset = Report(metrics=[
        RegressionPreset(
            mae_tests=[lt(0.3)],
            mean_error_tests=[gt(-0.2), lt(0.2)],
            rmse_tests=[lt(0.3)],
            r2score_tests=[gt(0.5)],
        )
    ])

    regression_snapshot_with_reference = regression_preset.run(current_data=current_dataset, reference_data=reference_dataset)

    # Drift report
    value_drift_columns = [ValueDrift(column=col, method="psi", threshold=0.05) for col in numerical_features]
    drift_report = Report([
        DriftedColumnsCount(
            cat_stattest="psi", num_stattest="wasserstein", 
            per_column_method={"target_regression":"psi", "predicted_regression":"psi"}, drift_share=0.8
        )
    ] + value_drift_columns, include_tests=False)

    drift_snapshot = drift_report.run(current_data=current_dataset, reference_data=reference_dataset)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    regression_snapshot_with_reference.save_html(REPORT_FEATURES_HTML)
    drift_snapshot.save_html(REPORT_DRIFT_HTML)
    
    return REPORT_FEATURES_HTML, REPORT_DRIFT_HTML

@app.get("/ping", response_class=PlainTextResponse)
def ping():
    return "evidently up\n"

@app.post("/generate")
def generate():
    try:
        path = build_report()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "report": path}

@app.get("/")
def index():
    if not os.path.exists(REPORT_FEATURES_HTML) and not os.path.exists(REPORT_DRIFT_HTML):
        try:
            build_report()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Generate failed: {e}")
    return FileResponse(REPORT_FEATURES_HTML, media_type="text/html")