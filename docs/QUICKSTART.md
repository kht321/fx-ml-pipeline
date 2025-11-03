# Quick Start Guide - S&P 500 ML Pipeline

## 🚀 Start All Services (Recommended)

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 8GB RAM minimum (16GB+ recommended)

### One Command to Start Everything:
```bash
# 1. Clone and setup
git clone https://github.com/kht321/fx-ml-pipeline.git
cd fx-ml-pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Optional: Configure email alerts
cp .env.monitoring.example .env.monitoring
# Edit .env.monitoring with your SMTP credentials

# 3. Start all services
docker-compose up -d
```

This starts 16 services:
1. **Airflow** (port 8080) - Workflow orchestration (training + inference DAGs)
2. **MLflow** (port 5001) - Experiment tracking and model registry
3. **FastAPI** (port 8000) - ML prediction REST API
4. **Streamlit** (port 8501) - Interactive dashboard
5. **PostgreSQL** - Metadata storage
6. **Redis** - Feature caching
7. **Nginx** - Load balancer
8. **Evidently** (port 8050) - Drift monitoring
9. **Model Servers** (Blue/Green deployment)

### Stop All Services:
```bash
docker-compose down

# Clean slate (remove volumes)
docker-compose down -v
```

---

## 📊 Access the Services

### 1. Airflow Web UI
**URL:** http://localhost:8080
**Credentials:** admin / admin

**Available DAGs:**
- `sp500_ml_pipeline_v4_docker` - Full training pipeline (17 tasks)
- `sp500_ml_pipeline_v4_docker_DEBUG` - Debug/testing DAG (16 tasks)
- `online_inference_dag` - Hourly real-time inference

**Quick Actions:**
```bash
# Trigger training pipeline
# 1. Go to http://localhost:8080
# 2. Click on "sp500_ml_pipeline_v4_docker"
# 3. Click "Trigger DAG" button (▶)

# View logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-webserver
```

### 2. MLflow Tracking UI
**URL:** http://localhost:5001

**Features:**
- Experiment tracking for XGBoost, LightGBM, AR (AutoRegressive OLS)
- Model comparison and metrics visualization
- Model registry with versioning
- Stage promotion (None → Staging → Production)
- Model artifacts (plots, reports, pickled models)

### 3. Streamlit Dashboard
**URL:** http://localhost:8501

**Features:**
- Real-time S&P 500 price predictions
- News sentiment timeline with FinBERT scores
- Feature importance visualization
- Model performance metrics
- Forecast vs actual comparison

### 4. FastAPI Prediction Service
**URL:** http://localhost:8000/docs (Swagger UI)

**API Usage:**
```bash
# Get prediction
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'

# Health check
curl http://localhost:8000/health

# Prediction history
curl http://localhost:8000/predictions/history

# Recent news
curl http://localhost:8000/news/recent
```

### 5. Evidently Monitoring
**URL:** http://localhost:8050

**Features:**
- Data drift detection reports
- Model performance degradation tracking
- Feature distribution comparison
- HTML reports with interactive visualizations

---

## 🧪 Test the Complete Pipeline

### Step 1: Start all services
```bash
docker-compose up -d
```

Wait 2-3 minutes for all services to initialize.

### Step 2: Verify services are running
```bash
docker-compose ps

# Check Airflow
curl http://localhost:8080/health

# Check MLflow
curl http://localhost:5001/health

# Check FastAPI
curl http://localhost:8000/health
```

### Step 3: Trigger Training Pipeline
1. Go to http://localhost:8080 (login: admin/admin)
2. Click on `sp500_ml_pipeline_v4_docker`
3. Click the ▶ "Trigger DAG" button
4. Monitor progress in the "Graph" view

**Pipeline Stages (17 tasks, ~25-35 min total):**
```
1. validate_bronze_data (30s)
2. silver_processing (3 parallel tasks: technical, microstructure, volatility) (5-8 min)
3. gold_processing (3 tasks: market merge, news signals, labels) (15-20 min)
4. validate_gold_data_quality (30s)
5. train_models (3 parallel: XGBoost, LightGBM, AR) (8-10 min)
6. select_best_model (10s)
7. deploy_model_to_production (30s)
```

### Step 4: View Training Results in MLflow
1. Go to http://localhost:5001
2. Click "Experiments" → "SP500_Training_v4"
3. Compare metrics across XGBoost, LightGBM, AR (AutoRegressive OLS)
4. View model artifacts (plots, metrics, pickled models)

### Step 5: Get Prediction via API
```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'
```

Expected response:
```json
{
  "timestamp": "2025-11-03T10:45:00",
  "instrument": "SPX500_USD",
  "prediction": 5234.56,
  "direction": "up",
  "confidence": 0.7891,
  "features_used": 114,
  "model_version": "lightgbm_v4_production",
  "model_type": "lightgbm",
  "test_rmse": 0.0229,
  "latency_ms": 32
}
```

### Step 6: View Dashboard
Go to http://localhost:8501 to see:
- Real-time predictions
- News sentiment timeline
- Feature importance
- Model performance metrics

---

## 📁 Manual Service Control

If you need to restart individual services:

```bash
# Restart Airflow scheduler
docker-compose restart airflow-scheduler

# Restart MLflow
docker-compose restart mlflow

# Restart FastAPI
docker-compose restart fastapi

# View logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f mlflow
```

---

## 🔍 View Logs

```bash
# View all service logs
docker-compose logs -f

# Individual service logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-webserver
docker-compose logs -f mlflow
docker-compose logs -f fastapi
docker-compose logs -f streamlit

# View Airflow task logs
# Go to Airflow UI → DAGs → Task → Logs
```

---

## 🐛 Troubleshooting

### Port 5000 Conflict (macOS)
**Problem:** macOS ControlCenter uses port 5000 for AirPlay

**Solution:** MLflow configured to use port 5001
```bash
open http://localhost:5001  # Use this instead of 5000
```

### Docker Services Not Starting
```bash
# Check Docker is running
docker info

# Check service status
docker-compose ps

# Restart all services
docker-compose down
docker-compose up -d

# Check logs for errors
docker-compose logs
```

### Airflow Tasks Failing (OOM)
**Problem:** Docker containers killed (exit 137)

**Solution:** Increase Docker Desktop memory
```bash
# Docker Desktop → Settings → Resources → Memory: 8GB+
```

### Port Already in Use
```bash
# Check what's using a port
lsof -ti:8080  # Airflow
lsof -ti:5001  # MLflow
lsof -ti:8000  # FastAPI
lsof -ti:8501  # Streamlit

# Kill process on specific port
lsof -ti:8080 | xargs kill -9
```

### Training Pipeline Fails
```bash
# Check Airflow scheduler logs
docker-compose logs -f airflow-scheduler

# Check data exists
ls -lh data_clean/bronze/market/
ls -lh data_clean/bronze/news/

# Verify bronze data validation
docker-compose exec airflow-scheduler python -c "
import pandas as pd
df = pd.read_json('data_clean/bronze/market/spx500_usd_m1.ndjson', lines=True)
print(f'✓ Market data: {len(df):,} rows')
"
```

### FinBERT Processing Too Slow
**Problem:** News sentiment analysis taking hours

**Solution:** Already optimized with batch processing
- Batch size: 64 articles
- 20-30x speedup vs sequential
- GPU auto-detected if available

### Email Alerts Not Sending
**Problem:** SMTP authentication failures

**Solution:**
1. Use app-specific password for Gmail
2. Generate at: https://myaccount.google.com/apppasswords
3. Update `.env.monitoring`
4. Test: `python -m src_clean.monitoring.email_alerter --to your_email@example.com`

---

## 📚 Additional Resources

- **Main README:** [README.md](../README.md)
- **Implementation Summary:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Airflow Setup:** [AIRFLOW_SETUP_GUIDE.md](AIRFLOW_SETUP_GUIDE.md)
- **Gmail Alerts:** [GMAIL_APP_PASSWORD_SETUP.md](GMAIL_APP_PASSWORD_SETUP.md)
- **Drift Monitoring:** [DRIFT_MONITORING.md](DRIFT_MONITORING.md)
- **API Docs:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5001
- **Airflow UI:** http://localhost:8080

---

## ✨ Key Features

### 1. Multi-Model Selection
- **3 Models Compete:** XGBoost, LightGBM, AR (AutoRegressive OLS)
- **Automatic Selection:** Best model by test RMSE
- **2-Stage Optuna Tuning:** Coarse search → Fine tuning
- **OOT2 Validation:** 10k most recent rows

### 2. Advanced Feature Engineering (114 features)
- **Technical Indicators (21):** RSI, MACD, Bollinger Bands, SMA, EMA
- **Returns (14):** Multi-timeframe (1-360 minutes)
- **Volatility (13):** GK, Parkinson, Yang-Zhang, EWMA
- **News Signals (22):** FinBERT sentiment analysis
- **Volume (7):** Liquidity and flow indicators
- **Microstructure (6):** Market impact, spread, order flow

### 3. Production MLOps Stack
- **Airflow:** Workflow orchestration (training + inference DAGs)
- **MLflow:** Experiment tracking + model registry
- **Evidently AI:** Drift detection + monitoring
- **Feast:** Feature store (online/batch serving)
- **Docker:** 16 services with Blue/Green deployment

### 4. Data Pipeline (Medallion Architecture)
```
Bronze (Raw)   → Silver (Features) → Gold (Training-Ready)
OANDA + GDELT  → 114 features      → Labels + Splits
1.7M candles   → Tech + News       → Train/Val/Test/OOT
25K+ articles  → FinBERT batch     → Reproducible splits
```

---

## 🎯 Next Steps

1. **Monitor Training:** Watch Airflow DAG execution at http://localhost:8080
2. **Compare Models:** View MLflow experiments at http://localhost:5001
3. **Test Predictions:** Use FastAPI at http://localhost:8000/docs
4. **View Dashboard:** Open Streamlit at http://localhost:8501
5. **Setup Alerts:** Configure `.env.monitoring` for email notifications
6. **Schedule Inference:** Enable `online_inference_dag` in Airflow
7. **Monitor Drift:** Check Evidently reports at http://localhost:8050

---

**Need Help?**
- Check [README.md](../README.md) for comprehensive documentation
- View logs: `docker-compose logs -f`
- GitHub Issues: https://github.com/kht321/fx-ml-pipeline/issues
