import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently import ColumnMapping

# USE CONTAINER PATHS (compose mounts these)
DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/data")
REPORTS_DIR = os.getenv("REPORTS_DIR", "/opt/airflow/reports")

GOLD = os.path.join(DATA_DIR, "gold", "features.csv")               # user_id, feat_mean, label
PREDS = os.path.join(DATA_DIR, "predictions", "batch_scores.csv")   # user_id, score
REPORT_HTML = os.path.join(REPORTS_DIR, "latest_report.html")

app = FastAPI(title="Evidently 0.6.7 Monitor")

# Ensure the reports dir exists before mounting as static
os.makedirs(REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

def build_report() -> str:
    if not (os.path.exists(GOLD) and os.path.exists(PREDS)):
        raise FileNotFoundError("Need gold/features.csv and predictions/batch_scores.csv")

    df_gold = pd.read_csv(GOLD)
    df_pred = pd.read_csv(PREDS)
    df = df_gold.merge(df_pred, on="user_id", how="inner")

    cm = ColumnMapping()
    cm.target = "label"
    cm.prediction = "score"
    cm.numerical_features = ["feat_mean"]
    cm.id = "user_id"

    report = Report(metrics=[ClassificationPreset()])
    report.run(current_data=df, reference_data=None, column_mapping=cm)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report.save_html(REPORT_HTML)
    return REPORT_HTML

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
    if not os.path.exists(REPORT_HTML):
        try:
            build_report()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Generate failed: {e}")
    return FileResponse(REPORT_HTML, media_type="text/html")