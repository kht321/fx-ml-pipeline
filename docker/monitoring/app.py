import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from evidently.report import Report
from evidently.metric_preset import RegressionPreset, DataDriftPreset
from evidently import ColumnMapping

# USE CONTAINER PATHS (compose mounts these)
DATA_DIR = os.getenv("DATA_DIR", "/data")
REPORTS_DIR = os.getenv("REPORTS_DIR", "/reports")

PREDICTIONS_LOG = os.path.join(DATA_DIR, "clean", "predictions", "prediction_log.jsonl")
REPORT_HTML = os.path.join(REPORTS_DIR, "latest_report.html")

app = FastAPI(title="Evidently Regression Monitor - S&P 500 Predictions")

# Ensure the reports dir exists before mounting as static
os.makedirs(REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")


def load_prediction_logs(min_records=10):
    """Load prediction logs from JSONL file."""
    if not os.path.exists(PREDICTIONS_LOG):
        raise FileNotFoundError(f"Prediction log not found: {PREDICTIONS_LOG}")

    predictions = []
    with open(PREDICTIONS_LOG, 'r') as f:
        for line in f:
            try:
                predictions.append(json.loads(line))
            except:
                pass

    if len(predictions) < min_records:
        raise ValueError(f"Need at least {min_records} predictions, found {len(predictions)}")

    # Convert to DataFrame
    df = pd.DataFrame(predictions)

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract feature columns (exclude metadata columns)
    metadata_cols = ['timestamp', 'instrument', 'prediction', 'probability', 'confidence',
                     'signal_strength', 'predicted_price', 'predicted_relative_change',
                     'task', 'model_version', 'features_used']

    # Feature columns are all numeric columns not in metadata
    feature_cols = []
    for col in df.columns:
        if col not in metadata_cols:
            # Handle list values (convert to scalar)
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    feature_cols.append(col)
                except:
                    pass
            elif pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

    return df, feature_cols


def build_report() -> str:
    """Build Evidently monitoring report for regression model."""
    try:
        # Load prediction logs
        df, feature_cols = load_prediction_logs(min_records=10)

        # Split into reference (older) and current (recent) data for drift detection
        split_point = int(len(df) * 0.7)
        reference_data = df.iloc[:split_point].copy()
        current_data = df.iloc[split_point:].copy()

        # Prepare column mapping for Evidently
        cm = ColumnMapping()
        cm.prediction = 'predicted_price'
        cm.datetime = 'timestamp'

        # Filter out empty/constant features
        valid_features = []
        for feat in feature_cols:
            if (not current_data[feat].isna().all() and
                not reference_data[feat].isna().all() and
                current_data[feat].nunique() > 1):
                valid_features.append(feat)

        # Include valid features for drift detection
        if valid_features:
            cm.numerical_features = valid_features

        # For prediction monitoring, we only use prediction drift since we don't have ground truth
        from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
        from evidently.metrics import ColumnDriftMetric, ColumnSummaryMetric

        # Build metrics list dynamically - focus on prediction monitoring
        metrics_list = [
            ColumnSummaryMetric(column_name='predicted_price'),
            ColumnDriftMetric(column_name='predicted_price'),
        ]

        # Add drift metrics for valid features only
        for feat in valid_features[:5]:  # Monitor top 5 valid features
            metrics_list.append(ColumnDriftMetric(column_name=feat))

        # Add dataset-level drift only if we have valid features
        if len(valid_features) > 0:
            metrics_list.insert(0, DatasetDriftMetric())
            metrics_list.append(DatasetMissingValuesMetric())

        # Create report with drift monitoring
        report = Report(metrics=metrics_list)

        # Run the report
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=cm
        )

        # Save HTML report
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report.save_html(REPORT_HTML)

        return REPORT_HTML

    except Exception as e:
        raise Exception(f"Report generation failed: {str(e)}")


@app.get("/ping", response_class=PlainTextResponse)
def ping():
    return "evidently regression monitor up\n"


@app.get("/stats", response_class=PlainTextResponse)
def stats():
    """Get quick stats about prediction logs."""
    try:
        df, feature_cols = load_prediction_logs(min_records=1)

        stats_text = f"""Prediction Log Statistics:
Total Predictions: {len(df)}
Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}
Features Tracked: {len(feature_cols)}

Recent Predictions:
- Bullish: {sum(df['prediction'] == 'bullish')}
- Bearish: {sum(df['prediction'] == 'bearish')}

Average Predicted Price: ${df['predicted_price'].mean():.2f}
Price Range: ${df['predicted_price'].min():.2f} - ${df['predicted_price'].max():.2f}
"""
        return stats_text
    except Exception as e:
        return f"Error getting stats: {str(e)}\n"


@app.post("/generate")
def generate():
    """Generate a new Evidently report."""
    try:
        path = build_report()
        return {"status": "ok", "report": path, "message": "Report generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def index():
    """Serve the latest report or generate a new one."""
    if not os.path.exists(REPORT_HTML):
        try:
            build_report()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Generate failed: {e}")
    return FileResponse(REPORT_HTML, media_type="text/html")
