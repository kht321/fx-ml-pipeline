# MLOps Infrastructure Plan

> **STATUS**: 📋 PLANNED (Not Yet Implemented)
> This document outlines the proposed MLOps infrastructure for productionizing the dual medallion FX prediction pipeline. Components described here are design proposals and have not been built yet.

---

## 🎯 Overview

This plan extends the current dual medallion architecture ([README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md)) with production-grade MLOps infrastructure:

- **Feast** for offline training and online feature serving
- **Airflow** for orchestration and scheduling
- **MLflow** for model registry and experiment tracking
- **FastAPI** for real-time inference serving
- **Prometheus + Grafana** for monitoring
- **Docker** for containerized local deployment

**Key Constraint**: Everything runs **locally via Docker Compose** (no cloud dependencies).

---

## 🏗️ Proposed Architecture

### **Current State (Implemented)**

```
Data Sources (OANDA API + News Scrapers)
    ↓
Bronze Layer (NDJSON files)
    ↓
Market Pipeline: Bronze → Silver (3 CSVs) → Gold
News Pipeline: Bronze → Silver (FinGPT) → Gold
    ↓
Combined Training (as-of join + lag features)
    ↓
XGBoost Training (manual trigger)
```

### **Target State (This Plan)**

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION (Airflow DAGs)                    │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────┐                        │
│  │ OANDA Collector │    │ News Scraper     │                        │
│  │ (Hourly Candles)│    │ (4 Sources)      │                        │
│  └────────┬────────┘    └────────┬─────────┘                        │
│           └──────────────────────┼─────────────────────┐            │
│                                  ↓                     ↓            │
│                          Bronze Layer (NDJSON)                       │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING (Airflow DAGs)                      │
├─────────────────────────────────────────────────────────────────────┤
│  Market Pipeline              News Pipeline                          │
│  Bronze → Silver → Gold       Bronze → Silver (FinGPT) → Gold       │
│           ↓                            ↓                             │
│  ┌────────────────────────────────────────────────────┐             │
│  │         Feast Offline Store (Historical)           │             │
│  │  - market_features (Gold layer)                    │             │
│  │  - news_features (Gold layer)                      │             │
│  │  - combined_features (as-of join + lags)           │             │
│  └────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                MODEL TRAINING (Airflow DAG)                          │
├─────────────────────────────────────────────────────────────────────┤
│  1. Feast.get_historical_features() → Training Dataset              │
│  2. Train XGBoost (multi-seed CV)                                   │
│  3. Evaluate (accuracy, calibration, PnL simulation)                │
│  4. Log to MLflow (metrics, params, artifacts)                      │
│  5. Register in MLflow Model Registry                               │
│     - Staging → Production promotion workflow                       │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│              FEATURE MATERIALIZATION (Airflow DAG)                   │
├─────────────────────────────────────────────────────────────────────┤
│  Feast.materialize_incremental()                                    │
│  - Offline Store → Online Store (Redis)                             │
│  - Triggered after each Silver → Gold transformation                │
│  - Keeps latest features in Redis for sub-ms lookup                 │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   INFERENCE SERVING (FastAPI)                        │
├─────────────────────────────────────────────────────────────────────┤
│  FastAPI Service                                                    │
│  ├─ Endpoint: POST /predict                                         │
│  │   Input: {"instrument": "USD_SGD", "timestamp": "..."}           │
│  │   Process:                                                       │
│  │   1. Feast.get_online_features() → Latest features from Redis   │
│  │   2. Load model from MLflow Model Registry (Production)          │
│  │   3. model.predict(features) → Prediction                        │
│  │   4. Log prediction metadata for monitoring                      │
│  │   Output: {"prediction": 0.78, "confidence": 0.82}               │
│  │                                                                  │
│  ├─ Endpoint: GET /health                                           │
│  │   Check: Redis, MLflow, model loaded                             │
│  │                                                                  │
│  └─ Endpoint: POST /feedback                                        │
│      Store: Realized outcomes for model retraining                  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      MONITORING (Prometheus + Grafana)               │
├─────────────────────────────────────────────────────────────────────┤
│  Data Quality Monitoring                                            │
│  ├─ Data freshness (time since last Bronze update)                  │
│  ├─ Schema validation (Bronze → Silver compatibility)               │
│  ├─ Feature distribution drift (Silver/Gold stats)                  │
│  └─ Missing value rates                                             │
│                                                                     │
│  Model Performance Monitoring                                       │
│  ├─ Prediction drift (distribution shift over time)                 │
│  ├─ Calibration (predicted probs vs realized outcomes)              │
│  ├─ Accuracy/F1/AUC (rolling window)                                │
│  ├─ Feature importance drift                                        │
│  └─ Brier score, log loss                                           │
│                                                                     │
│  Business Metrics Monitoring                                        │
│  ├─ Simulated PnL (paper trading)                                   │
│  ├─ Sharpe ratio, max drawdown                                      │
│  ├─ Trading signals generated per hour                              │
│  └─ Signal quality score                                            │
│                                                                     │
│  Infrastructure Monitoring                                          │
│  ├─ FastAPI latency (p50, p95, p99)                                 │
│  ├─ Feast online query latency                                      │
│  ├─ FinGPT GPU utilization                                          │
│  ├─ Redis memory usage                                              │
│  ├─ Airflow DAG run success/failure rates                           │
│  └─ Docker container health                                         │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION (Airflow DAGs)                        │
├─────────────────────────────────────────────────────────────────────┤
│  DAG 1: Data Collection (Every 10 min)                              │
│  ├─ Collect OANDA candles → Bronze                                  │
│  ├─ Scrape news → Bronze                                            │
│  └─ Trigger: Cron schedule                                          │
│                                                                     │
│  DAG 2: Market Feature Pipeline (Every 1 hour)                      │
│  ├─ Bronze → Silver (technical, microstructure, volatility)         │
│  ├─ Silver → Gold (merge + cross-instrument features)               │
│  ├─ Materialize to Feast Online Store                               │
│  └─ Trigger: Data freshness sensor                                  │
│                                                                     │
│  DAG 3: News Feature Pipeline (Every 30 min)                        │
│  ├─ Bronze → Silver (FinGPT sentiment analysis)                     │
│  ├─ Silver → Gold (aggregation + quality scoring)                   │
│  ├─ Materialize to Feast Online Store                               │
│  └─ Trigger: New articles in Bronze                                 │
│                                                                     │
│  DAG 4: Model Training (Daily 00:00 SGT)                            │
│  ├─ Fetch historical features from Feast Offline Store              │
│  ├─ Train XGBoost (multi-seed CV)                                   │
│  ├─ Evaluate on validation set                                      │
│  ├─ Log to MLflow (metrics, params, model)                          │
│  ├─ If performance > threshold: Promote to Staging                  │
│  └─ If Staging validation passes: Promote to Production             │
│                                                                     │
│  DAG 5: Model Validation (After training)                           │
│  ├─ Load Staging model from MLflow                                  │
│  ├─ Run backtests (PnL, Sharpe, turnover)                           │
│  ├─ Calibration checks (Brier score, ECE)                           │
│  ├─ Feature importance analysis (consistency check)                 │
│  ├─ If all checks pass: Promote to Production                       │
│  └─ Notify: Slack/Email with validation results                     │
│                                                                     │
│  DAG 6: Monitoring & Alerting (Every 15 min)                        │
│  ├─ Check data freshness (alert if >2 hours stale)                  │
│  ├─ Check prediction drift (alert if distribution shift)            │
│  ├─ Check model performance (alert if accuracy drops >5%)           │
│  ├─ Check infrastructure health (Redis, FastAPI, FinGPT)            │
│  └─ Send alerts to monitoring dashboard                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Component Details

### **1. Feast Feature Store**

#### **Purpose**
- **Offline Store**: Historical features for training (read from Gold layers)
- **Online Store**: Low-latency feature serving for inference (Redis)

#### **Proposed Configuration**

**Feature Repository Structure**:
```
feast_repo/
├── feature_store.yaml          # Feast configuration
├── features/
│   ├── market_features.py      # Market Gold feature definitions
│   ├── news_features.py        # News Gold feature definitions
│   └── combined_features.py    # Derived features (lags, interactions)
└── data_sources/
    ├── market_gold.py          # Point to data/market/gold/training/
    └── news_gold.py            # Point to data/news/gold/news_signals/
```

**`feature_store.yaml`** (Offline: Parquet, Online: Redis):
```yaml
project: fx-ml-pipeline
registry: data/feast/registry.db
provider: local
online_store:
  type: redis
  connection_string: "redis://redis:6379"
offline_store:
  type: file  # Read from Gold layer CSVs converted to Parquet
entity_key_serialization_version: 2
```

**`features/market_features.py`**:
```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, String, UnixTimestamp
from datetime import timedelta

# Entity: Currency pair
instrument = Entity(
    name="instrument",
    value_type=String,
    description="Currency pair (e.g., USD_SGD)"
)

# Data source: Market Gold layer
market_gold_source = FileSource(
    path="data/market/gold/training/market_features.parquet",
    timestamp_field="time",
    created_timestamp_column="created_at"
)

# Feature View: Market features
market_features = FeatureView(
    name="market_features",
    entities=["instrument"],
    ttl=timedelta(hours=2),  # Features valid for 2 hours
    schema=[
        Field(name="mid", dtype=Float64),
        Field(name="ret_1", dtype=Float64),
        Field(name="ret_5", dtype=Float64),
        Field(name="vol_20", dtype=Float64),
        Field(name="spread_pct", dtype=Float64),
        Field(name="zscore_20", dtype=Float64),
        Field(name="ewma_short", dtype=Float64),
        Field(name="ewma_long", dtype=Float64),
        Field(name="high_vol_regime", dtype=Float64),
        Field(name="asian_session", dtype=Float64),
        # ... all Market Gold features
    ],
    online=True,
    source=market_gold_source,
    tags={"team": "market-data"}
)
```

**`features/news_features.py`**:
```python
from feast import FeatureView, Field, FileSource
from feast.types import Float64, String
from datetime import timedelta

# Data source: News Gold layer
news_gold_source = FileSource(
    path="data/news/gold/news_signals/news_features.parquet",
    timestamp_field="time",
    created_timestamp_column="created_at"
)

# Feature View: News sentiment features
news_features = FeatureView(
    name="news_features",
    entities=["instrument"],
    ttl=timedelta(hours=6),  # News features valid for 6 hours
    schema=[
        Field(name="sentiment_score", dtype=Float64),
        Field(name="sgd_directional_signal", dtype=Float64),
        Field(name="confidence", dtype=Float64),
        Field(name="policy_implications", dtype=String),
        Field(name="market_coherence", dtype=String),
        Field(name="signal_strength_adjusted", dtype=Float64),
        # ... all News Gold features
    ],
    online=True,
    source=news_gold_source,
    tags={"team": "news-nlp"}
)
```

**`features/combined_features.py`**:
```python
from feast import FeatureView, Field, OnDemandFeatureView
from feast.types import Float64

# On-Demand Feature View: Compute lag features at query time
@on_demand_feature_view(
    sources=[market_features, news_features],
    schema=[
        Field(name="ret_1_lag1", dtype=Float64),
        Field(name="ret_1_lag2", dtype=Float64),
        Field(name="sentiment_lag1", dtype=Float64),
        # ... lag features
    ]
)
def lag_features(inputs: pd.DataFrame) -> pd.DataFrame:
    # Compute lags at inference time
    output = pd.DataFrame()
    output["ret_1_lag1"] = inputs["ret_1"].shift(1)
    output["ret_1_lag2"] = inputs["ret_1"].shift(2)
    output["sentiment_lag1"] = inputs["sentiment_score"].shift(1)
    return output
```

#### **Offline Training Workflow**
```python
# In training script (Airflow DAG)
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feast_repo/")

# Define entity dataframe (timestamps + instruments for training)
entity_df = pd.DataFrame({
    "time": pd.date_range("2025-01-01", "2025-12-31", freq="1H"),
    "instrument": "USD_SGD"
})

# Fetch historical features
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "market_features:ret_1",
        "market_features:vol_20",
        "news_features:sentiment_score",
        "news_features:sgd_directional_signal",
        "lag_features:ret_1_lag1",
        # ... all training features
    ]
).to_df()

# Add target variable (next-hour return direction)
training_data["y"] = (training_data["ret_1"].shift(-1) > 0).astype(int)

# Train XGBoost
import xgboost as xgb
model = xgb.XGBClassifier(**params)
model.fit(training_data.drop("y", axis=1), training_data["y"])
```

#### **Online Inference Workflow**
```python
# In FastAPI service
from feast import FeatureStore

store = FeatureStore(repo_path="feast_repo/")

# Get latest features for inference
features = store.get_online_features(
    features=[
        "market_features:ret_1",
        "market_features:vol_20",
        "news_features:sentiment_score",
        "lag_features:ret_1_lag1",
    ],
    entity_rows=[{"instrument": "USD_SGD"}]
).to_dict()

# Make prediction
prediction = model.predict(features)
```

#### **Materialization (Offline → Online)**
```python
# In Airflow DAG (after Gold layer update)
from feast import FeatureStore
from datetime import datetime, timedelta

store = FeatureStore(repo_path="feast_repo/")

# Materialize latest features to Redis
store.materialize_incremental(
    end_date=datetime.now()
)
# This copies latest Gold layer features → Redis for fast online lookup
```

---

### **2. Airflow Orchestration**

#### **Purpose**
- Schedule data collection, feature engineering, training, monitoring
- Manage dependencies between DAGs
- Retry logic and error handling

#### **Proposed DAG Structure**

**Directory Layout**:
```
airflow/
├── dags/
│   ├── data_collection.py          # Collect OANDA + news every 10 min
│   ├── market_feature_pipeline.py  # Bronze → Silver → Gold (hourly)
│   ├── news_feature_pipeline.py    # Bronze → Silver → Gold (30 min)
│   ├── model_training.py           # Daily training + MLflow logging
│   ├── model_validation.py         # Backtest + promote to production
│   └── monitoring.py               # Data/model/infra health checks
├── plugins/
│   ├── feast_operator.py           # Custom Feast materialization operator
│   ├── mlflow_operator.py          # MLflow model registry operator
│   └── slack_notifier.py           # Alert notifications
└── config/
    └── airflow.cfg
```

#### **DAG 1: Data Collection** (`data_collection.py`)
```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'fx-ml-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['alerts@fx-ml.local']
}

dag = DAG(
    'data_collection',
    default_args=default_args,
    description='Collect OANDA candles and news articles',
    schedule_interval='*/10 * * * *',  # Every 10 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False
)

collect_oanda = BashOperator(
    task_id='collect_oanda_candles',
    bash_command='python src/hourly_candle_collector.py --incremental',
    dag=dag
)

collect_news = BashOperator(
    task_id='collect_news',
    bash_command='python src/news_scraper.py --incremental',
    dag=dag
)

validate_bronze = BashOperator(
    task_id='validate_bronze_data',
    bash_command='python scripts/validate_bronze.py',
    dag=dag
)

# Dependencies
[collect_oanda, collect_news] >> validate_bronze
```

#### **DAG 2: Market Feature Pipeline** (`market_feature_pipeline.py`)
```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow_feast.operators import MaterializeFeaturesOperator
from datetime import datetime, timedelta

dag = DAG(
    'market_feature_pipeline',
    schedule_interval='0 * * * *',  # Every hour
    start_date=datetime(2025, 1, 1),
    catchup=False
)

# Wait for new Bronze data
wait_for_bronze = FileSensor(
    task_id='wait_for_new_bronze_data',
    filepath='data/bronze/prices/usd_sgd_hourly_2025.ndjson',
    poke_interval=60,
    timeout=600,
    dag=dag
)

# Bronze → Silver
build_silver = BashOperator(
    task_id='build_market_silver',
    bash_command='python src/build_market_features.py --incremental',
    dag=dag
)

# Silver → Gold
build_gold = BashOperator(
    task_id='build_market_gold',
    bash_command='python src/build_market_gold.py --incremental',
    dag=dag
)

# Convert CSV → Parquet for Feast
convert_to_parquet = BashOperator(
    task_id='convert_gold_to_parquet',
    bash_command='python scripts/csv_to_parquet.py --input data/market/gold/training/market_features.csv',
    dag=dag
)

# Materialize to Feast Online Store
materialize_feast = MaterializeFeaturesOperator(
    task_id='materialize_market_features',
    feature_views=['market_features'],
    feast_repo='feast_repo/',
    dag=dag
)

# Dependencies
wait_for_bronze >> build_silver >> build_gold >> convert_to_parquet >> materialize_feast
```

#### **DAG 3: News Feature Pipeline** (`news_feature_pipeline.py`)
```python
dag = DAG(
    'news_feature_pipeline',
    schedule_interval='*/30 * * * *',  # Every 30 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False
)

wait_for_news = FileSensor(
    task_id='wait_for_new_news',
    filepath='data/bronze/news/financial_news_2025.ndjson',
    poke_interval=60,
    timeout=1800,
    dag=dag
)

# Bronze → Silver (FinGPT analysis)
build_news_silver = BashOperator(
    task_id='build_news_silver_fingpt',
    bash_command='''
    python src/build_news_features.py \
        --use-fingpt \
        --use-market-context \
        --market-features-path data/market/silver/technical_features/sgd_vs_majors.csv \
        --incremental
    ''',
    dag=dag
)

# Silver → Gold
build_news_gold = BashOperator(
    task_id='build_news_gold',
    bash_command='python src/build_news_gold.py --incremental',
    dag=dag
)

convert_to_parquet = BashOperator(
    task_id='convert_news_gold_to_parquet',
    bash_command='python scripts/csv_to_parquet.py --input data/news/gold/news_signals/news_features.csv',
    dag=dag
)

materialize_feast = MaterializeFeaturesOperator(
    task_id='materialize_news_features',
    feature_views=['news_features'],
    feast_repo='feast_repo/',
    dag=dag
)

wait_for_news >> build_news_silver >> build_news_gold >> convert_to_parquet >> materialize_feast
```

#### **DAG 4: Model Training** (`model_training.py`)
```python
from airflow_mlflow.operators import LogModelOperator, RegisterModelOperator

dag = DAG(
    'model_training',
    schedule_interval='0 0 * * *',  # Daily at midnight SGT
    start_date=datetime(2025, 1, 1),
    catchup=False
)

# Fetch historical features from Feast Offline Store
fetch_training_data = BashOperator(
    task_id='fetch_feast_historical_features',
    bash_command='python scripts/fetch_training_data.py',
    dag=dag
)

# Train XGBoost with multi-seed CV
train_model = BashOperator(
    task_id='train_xgboost',
    bash_command='python src/train_combined_model.py --multi-seed-cv --n-seeds 5',
    dag=dag
)

# Evaluate on validation set
evaluate_model = BashOperator(
    task_id='evaluate_model',
    bash_command='python scripts/evaluate_model.py --split validation',
    dag=dag
)

# Log to MLflow
log_to_mlflow = LogModelOperator(
    task_id='log_model_to_mlflow',
    experiment_name='fx-prediction-usdsgd',
    model_path='models/xgboost_latest.pkl',
    artifacts=['feature_importance.png', 'confusion_matrix.png'],
    dag=dag
)

# Register in MLflow Model Registry (Staging)
register_model = RegisterModelOperator(
    task_id='register_model_staging',
    model_name='fx-usdsgd-predictor',
    model_uri='runs:/<run_id>/model',
    stage='Staging',
    dag=dag
)

fetch_training_data >> train_model >> evaluate_model >> log_to_mlflow >> register_model
```

#### **DAG 5: Model Validation & Promotion** (`model_validation.py`)
```python
dag = DAG(
    'model_validation',
    schedule_interval=None,  # Triggered by model_training DAG
    start_date=datetime(2025, 1, 1)
)

# Load Staging model from MLflow
load_staging_model = BashOperator(
    task_id='load_staging_model',
    bash_command='python scripts/load_mlflow_model.py --stage Staging',
    dag=dag
)

# Run backtests
backtest = BashOperator(
    task_id='run_backtest',
    bash_command='python scripts/backtest.py --model staging --metrics pnl,sharpe,drawdown',
    dag=dag
)

# Calibration checks
calibration_check = BashOperator(
    task_id='calibration_check',
    bash_command='python scripts/calibration.py --model staging',
    dag=dag
)

# Feature importance consistency check
feature_importance_check = BashOperator(
    task_id='feature_importance_check',
    bash_command='python scripts/check_feature_importance.py --compare-with production',
    dag=dag
)

# Promote to Production if all checks pass
promote_to_production = BashOperator(
    task_id='promote_to_production',
    bash_command='python scripts/promote_model.py --from Staging --to Production',
    trigger_rule='all_success',
    dag=dag
)

# Send notification
notify_slack = BashOperator(
    task_id='notify_slack',
    bash_command='python scripts/send_slack_notification.py --message "New model promoted to Production"',
    dag=dag
)

load_staging_model >> [backtest, calibration_check, feature_importance_check] >> promote_to_production >> notify_slack
```

#### **DAG 6: Monitoring** (`monitoring.py`)
```python
dag = DAG(
    'monitoring',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False
)

# Data freshness checks
check_data_freshness = BashOperator(
    task_id='check_data_freshness',
    bash_command='python scripts/check_data_freshness.py --alert-threshold 2h',
    dag=dag
)

# Prediction drift detection
check_prediction_drift = BashOperator(
    task_id='check_prediction_drift',
    bash_command='python scripts/check_drift.py --type prediction',
    dag=dag
)

# Model performance monitoring
check_model_performance = BashOperator(
    task_id='check_model_performance',
    bash_command='python scripts/check_model_performance.py --window 7d',
    dag=dag
)

# Infrastructure health
check_infrastructure = BashOperator(
    task_id='check_infrastructure_health',
    bash_command='python scripts/check_infra.py --services redis,fastapi,fingpt',
    dag=dag
)

[check_data_freshness, check_prediction_drift, check_model_performance, check_infrastructure]
```

---

### **3. MLflow Model Registry**

#### **Purpose**
- Track experiments (hyperparameters, metrics, artifacts)
- Version control for models
- Staging → Production promotion workflow

#### **Proposed Setup**

**MLflow Configuration**:
```yaml
# mlflow/config.yaml
backend_store_uri: sqlite:///mlflow/mlflow.db
default_artifact_root: mlflow/artifacts
serve:
  host: 0.0.0.0
  port: 5000
```

**Experiment Tracking** (in `src/train_combined_model.py`):
```python
import mlflow
import mlflow.xgboost

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("fx-prediction-usdsgd")

with mlflow.start_run(run_name=f"xgboost-{datetime.now():%Y%m%d-%H%M}"):
    # Log parameters
    mlflow.log_params({
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "lag_features": [1, 2, 3, 5, 10],
        "news_tolerance": "6H"
    })

    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Log metrics
    val_acc = accuracy_score(y_val, model.predict(X_val))
    mlflow.log_metrics({
        "val_accuracy": val_acc,
        "val_f1": f1_score(y_val, model.predict(X_val)),
        "val_auc": roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    })

    # Log feature importance plot
    plot_feature_importance(model)
    mlflow.log_artifact("feature_importance.png")

    # Log model
    mlflow.xgboost.log_model(model, "model")

    # Register model in MLflow Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "fx-usdsgd-predictor")
```

**Model Promotion Workflow**:
```python
# scripts/promote_model.py
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://mlflow:5000")

# Get latest Staging model
staging_models = client.get_latest_versions("fx-usdsgd-predictor", stages=["Staging"])
staging_model = staging_models[0]

# Run validation checks
if backtest_passed and calibration_passed:
    # Promote to Production
    client.transition_model_version_stage(
        name="fx-usdsgd-predictor",
        version=staging_model.version,
        stage="Production",
        archive_existing_versions=True  # Archive old Production model
    )
    print(f"Promoted model version {staging_model.version} to Production")
else:
    print("Validation checks failed. Model not promoted.")
```

**Loading Model in FastAPI**:
```python
# In FastAPI service
import mlflow.pyfunc

model_uri = "models:/fx-usdsgd-predictor/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Make predictions
prediction = model.predict(features)
```

---

### **4. FastAPI Inference Service**

#### **Purpose**
- REST API for real-time predictions
- Integrates with Feast Online Store and MLflow Model Registry
- Logging and monitoring hooks

#### **Proposed API Structure**

**Directory Layout**:
```
fastapi_service/
├── app/
│   ├── main.py                 # FastAPI app
│   ├── models.py               # Pydantic schemas
│   ├── inference.py            # Prediction logic
│   ├── monitoring.py           # Prometheus metrics
│   └── config.py               # Configuration
├── tests/
│   ├── test_api.py
│   └── test_inference.py
└── requirements.txt
```

#### **`app/main.py`**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore
import mlflow.pyfunc
import prometheus_client
from datetime import datetime
import logging

app = FastAPI(title="FX Prediction API", version="1.0.0")

# Initialize Feast and MLflow
feast_store = FeatureStore(repo_path="feast_repo/")
model = mlflow.pyfunc.load_model("models:/fx-usdsgd-predictor/Production")

# Prometheus metrics
prediction_counter = prometheus_client.Counter(
    'predictions_total',
    'Total number of predictions'
)
prediction_latency = prometheus_client.Histogram(
    'prediction_latency_seconds',
    'Prediction latency'
)

# Request schema
class PredictionRequest(BaseModel):
    instrument: str
    timestamp: str = None  # Optional, defaults to now

# Response schema
class PredictionResponse(BaseModel):
    instrument: str
    prediction: float  # Probability of price increase
    direction: str  # "bullish" or "bearish"
    confidence: float
    features_used: dict
    model_version: str
    timestamp: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Get real-time FX prediction for a currency pair.

    Example:
    POST /predict
    {
        "instrument": "USD_SGD",
        "timestamp": "2025-01-15T10:00:00Z"
    }
    """
    with prediction_latency.time():
        try:
            # Get latest features from Feast Online Store (Redis)
            features = feast_store.get_online_features(
                features=[
                    "market_features:ret_1",
                    "market_features:vol_20",
                    "market_features:spread_pct",
                    "news_features:sentiment_score",
                    "news_features:sgd_directional_signal",
                    "lag_features:ret_1_lag1",
                    # ... all training features
                ],
                entity_rows=[{"instrument": request.instrument}]
            ).to_dict()

            # Make prediction
            pred_proba = model.predict(features)[0]
            direction = "bullish" if pred_proba > 0.5 else "bearish"
            confidence = abs(pred_proba - 0.5) * 2  # Scale to 0-1

            # Increment counter
            prediction_counter.inc()

            # Log prediction for monitoring
            logging.info(f"Prediction: {request.instrument} @ {datetime.now()} -> {pred_proba:.4f}")

            return PredictionResponse(
                instrument=request.instrument,
                prediction=pred_proba,
                direction=direction,
                confidence=confidence,
                features_used=features,
                model_version=model.metadata.get_model_info().version,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """
    Health check endpoint.
    Checks: Redis (Feast), MLflow model loaded, feature availability
    """
    try:
        # Check Feast Online Store
        feast_store.get_online_features(
            features=["market_features:ret_1"],
            entity_rows=[{"instrument": "USD_SGD"}]
        )

        # Check model loaded
        assert model is not None

        return {
            "status": "healthy",
            "feast_online_store": "ok",
            "mlflow_model": "ok",
            "model_version": model.metadata.get_model_info().version
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/feedback")
async def feedback(instrument: str, timestamp: str, realized_direction: str):
    """
    Store realized outcomes for model retraining and monitoring.

    Example:
    POST /feedback
    {
        "instrument": "USD_SGD",
        "timestamp": "2025-01-15T10:00:00Z",
        "realized_direction": "bullish"  # Actual market outcome
    }
    """
    # Store in feedback database for retraining
    # This enables continuous learning and drift detection
    # TODO: Implement feedback storage
    return {"status": "feedback recorded"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return prometheus_client.generate_latest()
```

#### **Running FastAPI Service**
```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production (in Docker)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

### **5. Monitoring Infrastructure**

#### **Purpose**
- Track data quality, model performance, infrastructure health
- Alert on anomalies and degradation

#### **Proposed Monitoring Stack**

**Components**:
- **Prometheus**: Metrics collection (from FastAPI, Airflow, Redis, FinGPT)
- **Grafana**: Dashboards for visualization
- **Alertmanager**: Notification routing (Slack, email)

#### **Metrics to Track**

**Data Quality Metrics** (`scripts/check_data_freshness.py`):
```python
from prometheus_client import Gauge, push_to_gateway
import pandas as pd
from datetime import datetime, timedelta

# Prometheus metrics
data_freshness_gauge = Gauge('data_freshness_hours', 'Hours since last Bronze update', ['layer', 'pipeline'])
missing_value_rate = Gauge('missing_value_rate', 'Missing value percentage', ['feature'])
schema_validation_status = Gauge('schema_validation_status', 'Schema validation (1=pass, 0=fail)', ['layer'])

# Check Bronze layer freshness
def check_freshness():
    market_bronze = pd.read_json('data/bronze/prices/usd_sgd_hourly_2025.ndjson', lines=True)
    last_update = pd.to_datetime(market_bronze['time'].max())
    hours_stale = (datetime.now() - last_update).total_seconds() / 3600

    data_freshness_gauge.labels(layer='bronze', pipeline='market').set(hours_stale)

    if hours_stale > 2:
        send_alert(f"Market Bronze data is {hours_stale:.1f} hours stale!")

    # Push to Prometheus
    push_to_gateway('prometheus:9091', job='data_quality', registry=...)
```

**Model Performance Metrics** (`scripts/check_model_performance.py`):
```python
from prometheus_client import Gauge

# Prometheus metrics
model_accuracy = Gauge('model_accuracy', 'Rolling accuracy', ['window'])
model_calibration = Gauge('model_calibration_brier', 'Brier score')
prediction_drift = Gauge('prediction_drift_kl_divergence', 'KL divergence from training distribution')

# Check model performance
def check_model_performance():
    # Load predictions + realized outcomes from feedback database
    predictions = load_predictions(window='7d')
    realized = load_realized_outcomes(window='7d')

    # Accuracy
    acc = accuracy_score(realized, predictions > 0.5)
    model_accuracy.labels(window='7d').set(acc)

    # Calibration (Brier score)
    brier = brier_score_loss(realized, predictions)
    model_calibration.set(brier)

    # Drift detection (compare prediction distribution)
    training_preds = load_training_predictions()
    kl_div = kl_divergence(training_preds, predictions)
    prediction_drift.set(kl_div)

    if kl_div > 0.1:
        send_alert(f"Prediction drift detected! KL divergence: {kl_div:.4f}")
```

**Business Metrics** (`scripts/simulate_pnl.py`):
```python
from prometheus_client import Gauge

# Prometheus metrics
simulated_pnl = Gauge('simulated_pnl_usd', 'Paper trading PnL', ['strategy'])
sharpe_ratio = Gauge('sharpe_ratio', 'Rolling Sharpe ratio', ['window'])
max_drawdown = Gauge('max_drawdown_pct', 'Maximum drawdown percentage')

# Simulate trading based on predictions
def simulate_pnl():
    predictions = load_predictions(window='30d')
    realized_returns = load_realized_returns(window='30d')

    # Simple strategy: Long if pred > 0.6, Short if pred < 0.4
    positions = np.where(predictions > 0.6, 1, np.where(predictions < 0.4, -1, 0))
    pnl = (positions * realized_returns).sum()

    simulated_pnl.labels(strategy='threshold').set(pnl)

    # Sharpe ratio
    returns = positions * realized_returns
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    sharpe_ratio.labels(window='30d').set(sharpe)

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    max_drawdown.set(abs(max_dd))
```

**Infrastructure Metrics** (Auto-collected by Prometheus):
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['fastapi:8000']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'airflow'
    static_configs:
      - targets: ['airflow:8080']
    metrics_path: '/metrics'

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

#### **Grafana Dashboards**

**Dashboard 1: Data Quality**
```
Panels:
- Data Freshness (time series): hours since last Bronze update
- Missing Value Rates (heatmap): per feature
- Schema Validation Status (status panel): pass/fail
- Feature Distribution Drift (histogram): KL divergence
```

**Dashboard 2: Model Performance**
```
Panels:
- Accuracy (time series): rolling 7d/30d accuracy
- Calibration Curve (scatter): predicted probs vs realized outcomes
- Brier Score (gauge): current calibration quality
- Feature Importance Stability (bar chart): compare to training
```

**Dashboard 3: Business Metrics**
```
Panels:
- Simulated PnL (time series): cumulative PnL
- Sharpe Ratio (gauge): rolling 30d
- Max Drawdown (gauge): current maximum drawdown
- Trading Signals (time series): signals generated per hour
```

**Dashboard 4: Infrastructure**
```
Panels:
- FastAPI Latency (histogram): p50, p95, p99
- Feast Query Latency (time series): Redis response time
- FinGPT GPU Utilization (gauge): GPU memory/compute usage
- Redis Memory Usage (gauge): current memory consumption
- Airflow DAG Success Rate (time series): success/failure ratio
```

---

### **6. Docker Deployment**

#### **Purpose**
- Containerize all services for local orchestration
- Reproducible environment (no cloud dependencies)

#### **Proposed Docker Compose Architecture**

**Directory Layout**:
```
docker/
├── docker-compose.yml          # Main orchestration file
├── Dockerfile.fastapi          # FastAPI service
├── Dockerfile.airflow          # Airflow scheduler + webserver
├── Dockerfile.fingpt           # FinGPT inference service
└── .env.example                # Environment variables
```

#### **`docker-compose.yml`**
```yaml
version: '3.8'

services:
  # PostgreSQL for Airflow metadata
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 10s
      retries: 5

  # Redis for Feast Online Store
  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  # Airflow Scheduler
  airflow-scheduler:
    build:
      context: .
      dockerfile: Dockerfile.airflow
    depends_on:
      - postgres
      - redis
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__FERNET_KEY: ${AIRFLOW_FERNET_KEY}
    volumes:
      - ../airflow/dags:/opt/airflow/dags
      - ../airflow/plugins:/opt/airflow/plugins
      - ../data:/opt/airflow/data
      - ../src:/opt/airflow/src
      - ../feast_repo:/opt/airflow/feast_repo
    command: scheduler

  # Airflow Webserver
  airflow-webserver:
    build:
      context: .
      dockerfile: Dockerfile.airflow
    depends_on:
      - postgres
      - airflow-scheduler
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    ports:
      - "8080:8080"
    volumes:
      - ../airflow/dags:/opt/airflow/dags
      - ../airflow/plugins:/opt/airflow/plugins
    command: webserver

  # MLflow Tracking Server
  mlflow:
    image: python:3.10
    command: >
      bash -c "
        pip install mlflow &&
        mlflow server
          --backend-store-uri sqlite:///mlflow/mlflow.db
          --default-artifact-root /mlflow/artifacts
          --host 0.0.0.0
          --port 5000
      "
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow

  # FinGPT Inference Service (GPU)
  fingpt:
    build:
      context: .
      dockerfile: Dockerfile.fingpt
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ../src:/app/src
      - ../models:/app/models
    environment:
      CUDA_VISIBLE_DEVICES: 0
    command: python src/fingpt_processor.py --serve --port 8001

  # FastAPI Inference Service
  fastapi:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
    depends_on:
      - redis
      - mlflow
    ports:
      - "8000:8000"
    volumes:
      - ../fastapi_service:/app
      - ../feast_repo:/app/feast_repo
    environment:
      FEAST_REDIS_HOST: redis
      MLFLOW_TRACKING_URI: http://mlflow:5000
    command: gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ../monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    volumes:
      - ../monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ../monitoring/grafana/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
      - grafana-data:/var/lib/grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: false

  # Redis Exporter for Prometheus
  redis-exporter:
    image: oliver006/redis_exporter:latest
    depends_on:
      - redis
    ports:
      - "9121:9121"
    environment:
      REDIS_ADDR: redis:6379

volumes:
  postgres-data:
  redis-data:
  mlflow-data:
  prometheus-data:
  grafana-data:
```

#### **Dockerfile.fastapi**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY fastapi_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fastapi_service/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### **Dockerfile.airflow**
```dockerfile
FROM apache/airflow:2.7.0-python3.10

USER root
RUN apt-get update && apt-get install -y git

USER airflow

# Install additional Python packages
COPY airflow/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Feast, MLflow, project dependencies
RUN pip install feast mlflow xgboost pandas numpy scikit-learn
```

#### **Dockerfile.fingpt**
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip

WORKDIR /app

# Install FinGPT dependencies
COPY src/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install transformers torch accelerate bitsandbytes

EXPOSE 8001

CMD ["python3", "src/fingpt_processor.py", "--serve"]
```

#### **Running the Stack**
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f fastapi

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

#### **Accessing Services**
- **Airflow UI**: http://localhost:8080 (user: admin, password: admin)
- **MLflow UI**: http://localhost:5000
- **FastAPI Swagger**: http://localhost:8000/docs
- **Grafana Dashboards**: http://localhost:3000 (user: admin, password: admin)
- **Prometheus**: http://localhost:9090

---

## 🔄 Complete MLOps Workflow

### **Daily Operation Flow**

```
00:00 SGT - Model Training DAG Triggers
    ↓
1. Fetch historical features from Feast Offline Store
    ↓
2. Train XGBoost with multi-seed CV
    ↓
3. Log metrics/params to MLflow
    ↓
4. Register model as "Staging" in MLflow Model Registry
    ↓
5. Trigger Model Validation DAG
    ↓
6. Run backtests (PnL, Sharpe, calibration)
    ↓
7. If validation passes: Promote "Staging" → "Production"
    ↓
8. FastAPI automatically loads new "Production" model
    ↓
9. Send Slack notification: "New model v2.5 deployed to Production"

---

Every 10 minutes - Data Collection
    ↓
1. Collect OANDA candles → Bronze
2. Scrape news → Bronze
3. Validate schema

Every 1 hour - Market Feature Pipeline
    ↓
1. Bronze → Silver (technical, microstructure, volatility)
2. Silver → Gold (merge + cross-instrument features)
3. Convert Gold CSV → Parquet
4. Materialize to Feast Online Store (Redis)

Every 30 minutes - News Feature Pipeline
    ↓
1. Bronze → Silver (FinGPT sentiment analysis with market context)
2. Silver → Gold (aggregation + time decay)
3. Convert Gold CSV → Parquet
4. Materialize to Feast Online Store (Redis)

Every 15 minutes - Monitoring
    ↓
1. Check data freshness (alert if >2h stale)
2. Check prediction drift (KL divergence)
3. Check model accuracy (rolling 7d)
4. Check infrastructure (Redis, FastAPI, FinGPT GPU)
5. Push metrics to Prometheus
6. Alert if thresholds breached

Real-time - Inference Serving
    ↓
User/Frontend → POST /predict {"instrument": "USD_SGD"}
    ↓
1. FastAPI receives request
2. Feast.get_online_features() → Fetch from Redis (sub-ms)
3. Load MLflow "Production" model
4. model.predict(features)
5. Log prediction + metadata
6. Return {"prediction": 0.78, "confidence": 0.82}
```

---

## 📊 Expected Performance Characteristics

### **Latency Targets**

| Component | Operation | Target Latency | Notes |
|-----------|-----------|----------------|-------|
| **Feast Offline** | get_historical_features() | <10s | For 1 year hourly data (~8,760 rows) |
| **Feast Online** | get_online_features() | <10ms | Redis lookup for 50 features |
| **MLflow** | Load model | <500ms | Cached after first load |
| **FastAPI** | /predict endpoint | <100ms | End-to-end (Feast + inference) |
| **FinGPT** | Sentiment analysis | 2-5s | GPU inference (LLaMA2-7B) |
| **Airflow** | DAG execution | Varies | Market pipeline: ~5 min, Training: ~30 min |

### **Throughput Targets**

| Component | Metric | Target | Notes |
|-----------|--------|--------|-------|
| **FastAPI** | Predictions/sec | 100+ | With 4 workers |
| **Feast Materialization** | Features/sec | 1,000+ | Offline → Online (Redis) |
| **FinGPT** | Articles/hour | 20-50 | GPU-bound (single GPU) |
| **Airflow** | DAG concurrency | 10+ | Parallel pipeline execution |

---

## 🚨 Open Questions & Validation Needed

### **1. Feast Offline Store: File Format**
- **Question**: Can Feast's file-based offline store read directly from CSV, or must we convert to Parquet?
- **Current Plan**: Convert Gold layer CSVs → Parquet in Airflow DAG before materialization
- **Validation**: Test Feast FileSource with CSV vs Parquet performance

### **2. Lag Feature Engineering: Feast vs Pre-computation**
- **Question**: Should lag features be computed at query time (OnDemandFeatureView) or pre-computed in Gold layer?
- **Current Plan**: Pre-compute lags in Gold layer for training, use OnDemandFeatureView for inference
- **Trade-off**:
  - Pre-compute: Faster inference, more storage
  - On-demand: Flexible, but adds latency
- **Validation**: Benchmark both approaches

### **3. FinGPT GPU Requirements**
- **Question**: Can FinGPT run efficiently on CPU for inference if GPU unavailable?
- **Current Plan**: GPU required (CUDA), fallback to lexicon-based sentiment on failure
- **Validation**: Test 8-bit quantization on CPU, measure latency

### **4. Model Retraining Frequency**
- **Question**: Daily retraining may be overkill for FX data (slower regime changes than stocks)
- **Current Plan**: Daily training, but only promote if significant improvement
- **Alternative**: Weekly training, daily validation on Staging model
- **Validation**: A/B test daily vs weekly retraining impact on accuracy

### **5. Redis Memory Sizing**
- **Question**: How much Redis memory for 50 features × 10 instruments × 24 hours?
- **Estimate**: ~50 features × 10 instruments × 8 bytes × 24 hours = ~100 KB (negligible)
- **Buffer**: 1 GB Redis should be more than sufficient
- **Validation**: Monitor Redis memory usage in production

### **6. Feast Entity Key Design**
- **Question**: Should entity be `instrument` alone, or `(instrument, time)` tuple?
- **Current Plan**: `instrument` entity + timestamp field
- **Validation**: Confirm Feast temporal join behavior with entity-only vs compound key

### **7. MLflow Model Registry: Multi-Instrument Models**
- **Question**: One model per instrument (USD_SGD, EUR_USD) or single multi-instrument model?
- **Current Plan**: Single multi-instrument model with `instrument` as categorical feature
- **Trade-off**:
  - Single model: Easier management, learns cross-instrument patterns
  - Multiple models: Specialized per instrument, independent tuning
- **Validation**: Compare performance of both approaches

### **8. Monitoring Alert Thresholds**
- **Question**: What are sensible thresholds for data freshness, drift, accuracy drop?
- **Current Plan**:
  - Data freshness: >2 hours → alert
  - Prediction drift (KL divergence): >0.1 → alert
  - Accuracy drop: >5% relative drop vs training → alert
- **Validation**: Tune thresholds during pilot phase

---

## 🛠️ Implementation Roadmap

### **Phase 1: Core Infrastructure (Weeks 1-2)**
- [ ] Set up Docker Compose with all services
- [ ] Configure Feast feature repository
  - [ ] Define market_features, news_features FeatureViews
  - [ ] Set up Redis Online Store
  - [ ] Test Offline Store with Parquet files
- [ ] Set up MLflow tracking server
  - [ ] Configure PostgreSQL backend
  - [ ] Test experiment logging
- [ ] Basic FastAPI service
  - [ ] /predict endpoint (mock model)
  - [ ] /health endpoint
  - [ ] Prometheus metrics

### **Phase 2: Airflow Orchestration (Weeks 3-4)**
- [ ] Implement DAG 1: Data Collection
- [ ] Implement DAG 2: Market Feature Pipeline
  - [ ] Bronze → Silver → Gold transformations
  - [ ] CSV → Parquet conversion
  - [ ] Feast materialization operator
- [ ] Implement DAG 3: News Feature Pipeline
  - [ ] FinGPT integration in Docker
  - [ ] Market context injection
  - [ ] Feast materialization
- [ ] Test end-to-end data flow

### **Phase 3: Model Training & Registry (Week 5)**
- [ ] Implement DAG 4: Model Training
  - [ ] Feast historical feature retrieval
  - [ ] XGBoost training with MLflow logging
  - [ ] Model registration (Staging)
- [ ] Implement DAG 5: Model Validation
  - [ ] Backtest suite
  - [ ] Calibration checks
  - [ ] Promotion workflow (Staging → Production)
- [ ] Integrate MLflow model loading in FastAPI

### **Phase 4: Monitoring & Alerting (Week 6)**
- [ ] Set up Prometheus + Grafana
- [ ] Implement monitoring scripts
  - [ ] Data quality checks
  - [ ] Model performance metrics
  - [ ] Infrastructure health
- [ ] Create Grafana dashboards
- [ ] Configure Alertmanager (Slack notifications)
- [ ] Implement DAG 6: Monitoring

### **Phase 5: Testing & Validation (Week 7)**
- [ ] Integration tests (end-to-end pipeline)
- [ ] Performance benchmarks (latency, throughput)
- [ ] Load testing (FastAPI under concurrent requests)
- [ ] Failure recovery testing (kill containers, check retries)
- [ ] Documentation updates

### **Phase 6: Pilot Deployment (Week 8)**
- [ ] Deploy to local Docker environment
- [ ] Run pilot for 1 week with synthetic news data
- [ ] Monitor all metrics
- [ ] Tune alert thresholds
- [ ] Bug fixes and optimizations

---

## 📚 Technology Stack Summary

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Feature Store** | Feast 0.34+ | Offline training + Online serving |
| **Orchestration** | Apache Airflow 2.7+ | DAG scheduling, retry logic |
| **Model Registry** | MLflow 2.9+ | Experiment tracking, model versioning |
| **Inference API** | FastAPI 0.104+ | REST API, async serving |
| **Offline Storage** | Parquet (file-based) | Feast offline store backend |
| **Online Storage** | Redis 7.x | Feast online store (sub-ms lookup) |
| **Monitoring** | Prometheus + Grafana | Metrics, dashboards, alerts |
| **Message Queue** | (Future) Kafka/RabbitMQ | Real-time streaming (if needed) |
| **Containerization** | Docker + Docker Compose | Local deployment |
| **Model Framework** | XGBoost 2.0+ | Gradient boosting |
| **NLP Model** | FinGPT (LLaMA2-7B LoRA) | Sentiment analysis |
| **Database** | PostgreSQL 14 | Airflow metadata, MLflow backend |

---

## 🎯 Success Criteria

### **Technical Metrics**
- [ ] FastAPI /predict latency <100ms (p95)
- [ ] Feast online feature retrieval <10ms (p95)
- [ ] Airflow DAG success rate >95%
- [ ] Model training completes in <30 minutes
- [ ] Zero data loss during pipeline failures (Bronze immutability)

### **Business Metrics**
- [ ] Model accuracy: 78-85% (on validation set)
- [ ] Simulated PnL: Positive over 30-day window
- [ ] Sharpe ratio: >1.0 (risk-adjusted returns)
- [ ] Max drawdown: <15%

### **Operational Metrics**
- [ ] Data freshness: Always <2 hours
- [ ] Monitoring alerts: <5% false positive rate
- [ ] Model retraining: Automated daily with validation
- [ ] Docker stack: All services healthy 99%+ uptime

---

## 📝 Next Steps

1. **Review this plan** with the team for feedback
2. **Validate open questions** (especially Feast file format, lag feature strategy)
3. **Set up development environment** (Docker, Feast repo, Airflow)
4. **Begin Phase 1 implementation** (Core infrastructure)
5. **Document decisions** as implementation progresses

---

**This MLOps plan provides a production-grade infrastructure for the dual medallion FX prediction pipeline, with local Docker deployment and comprehensive monitoring.**
