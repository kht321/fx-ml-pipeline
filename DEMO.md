# FX ML Pipeline - Complete Demo Guide

## Table of Contents
- [Quick Start (5 Minutes)](#quick-start-5-minutes)
- [Full Docker Stack Demo](#full-docker-stack-demo)
- [Component-Specific Demos](#component-specific-demos)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (5 Minutes)

### Prerequisites
- Python 3.11+
- OANDA practice account ([Get free API credentials](https://www.oanda.com/us-en/trading/api/))
- 8GB RAM minimum

### Step 1: Environment Setup
```bash
# Clone repository and setup virtual environment
git clone https://github.com/kht321/fx-ml-pipeline.git
cd fx-ml-pipeline

# Create Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure OANDA Credentials
```bash
# Create .env file with your OANDA credentials
cat > .env << EOF
OANDA_TOKEN=your_practice_token_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENV=practice
EOF
```

### Step 3: Run the Pipeline
```bash
# Execute the full medallion pipeline (Bronze → Silver → Gold → Training)
python src_clean/run_full_pipeline.py \
  --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
  --bronze-news data_clean/bronze/news \
  --output-dir data_clean \
  --prediction-horizon 30
```

**Expected Duration:** 5-10 minutes

**What You'll See:**
```
[INFO] Stage 1/7: Processing market technical features...
[INFO] Stage 2/7: Processing market microstructure features...
[INFO] Stage 3/7: Processing market volatility features...
[INFO] Stage 4/7: Merging gold market features...
[INFO] Stage 5/7: Processing news sentiment...
[INFO] Stage 6/7: Generating FinBERT trading signals...
[INFO] Stage 7/7: Training XGBoost model with MLflow...
[SUCCESS] Pipeline completed! Model saved to data_clean/models/
```

### Step 4: View Results

**Option A: MLflow UI**
```bash
# Terminal 1: Start MLflow tracking server
mlflow ui --backend-store-uri file:./mlruns --port 5002 --host 0.0.0.0

# Access at: http://localhost:5002
```

**Option B: Streamlit Dashboard (Enhanced v2.0)**
```bash
# Terminal 2: Launch interactive dashboard
streamlit run src_clean/ui/streamlit_dashboard.py

# Access at: http://localhost:8501
```

**New Dashboard Features:**
- ✅ News snippet preview (200 chars) with full article expansion
- ✅ ML-based price forecast (dotted line with confidence bands)
- ✅ Sentiment timeline chart
- ✅ Automatic prediction history tracking (last 100)
- ✅ Recent news tab with filtering
- ✅ Forecast confidence metrics and risk indicators

**Outputs Created:**
```
data_clean/
├── silver/market/
│   ├── technical/spx500_technical.csv        (17 features)
│   ├── microstructure/spx500_microstructure.csv (7 features)
│   └── volatility/spx500_volatility.csv      (7 features)
├── gold/market/
│   ├── features/spx500_features.csv          (64 base features)
│   └── labels/spx500_labels_30min.csv        (binary targets)
├── gold/news/
│   └── signals/spx500_news_signals.csv       (6 FinBERT signals)
└── models/
    ├── xgboost_classification_enhanced.pkl   (best model)
    └── best_model_selection.json             (performance metrics)
```

---

## Full Docker Stack Demo

This comprehensive demo showcases all 16 Docker services running together.

### Prerequisites
- Docker Desktop running
- 16GB RAM
- 50GB free disk space
- OANDA API credentials configured in `.env`

### Step 1: Start All Services

```bash
# Recommended: Use the fixed startup script
# This handles port 5000 conflict automatically (MLflow → 5001)
./fix_port_and_start.sh
```

**What this script does:**
1. Checks if port 5000 is in use (macOS ControlCenter)
2. Stops any existing containers
3. Builds Airflow 2.9.3 image with correct dependencies
4. Starts infrastructure (PostgreSQL, Redis, MLflow on port 5001)
5. Initializes Airflow database
6. Starts all Airflow services (webserver, scheduler, dag-processor)
7. Waits for services to be fully ready

**Expected Duration:** 3-5 minutes (first time may take 10-15 minutes for image builds)

### Step 2: Verify Services

```bash
# Check all services are running
docker-compose ps

# Should show all services as "running" or "healthy":
# - ml-postgres (healthy)
# - ml-redis (healthy)
# - ml-mlflow (running) - Port 5001
# - ml-airflow-postgres (healthy)
# - ml-airflow-webserver (running) - Port 8080
# - ml-airflow-scheduler (running)
# - ml-airflow-dag-processor (running)
# - ml-fastapi (running) - Port 8000
# - ml-streamlit (running) - Port 8501
# - model-blue (running) - Port 8001
# - model-green (running) - Port 8002
# - ml-nginx (running) - Port 8088
# - ml-evidently (running) - Port 8050
```

### Step 3: Access Services

**All Service URLs:**

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Airflow** | http://localhost:8080 | admin / admin | Workflow orchestration, trigger DAGs |
| **MLflow** | http://localhost:5001 | None | Experiment tracking, model registry |
| **FastAPI** | http://localhost:8000/docs | None | REST API documentation |
| **Streamlit** | http://localhost:8501 | None | Interactive ML dashboard |
| **Evidently** | http://localhost:8050 | None | Model monitoring reports |
| **Nginx** | http://localhost:8088 | None | Load balancer (blue/green models) |

**Note:** MLflow is on **port 5001** (not 5000) to avoid macOS conflicts.

### Step 4: Trigger Production Pipeline

**Option A: Via Airflow UI (Recommended)**
```bash
# 1. Open Airflow UI
open http://localhost:8080

# 2. Login with: admin / admin

# 3. Find DAG: sp500_ml_pipeline_v3_production

# 4. Toggle switch to ON (enable DAG)

# 5. Click ▶️ button to trigger manually
```

**Option B: Via CLI**
```bash
# Trigger DAG from command line
docker-compose exec airflow-scheduler airflow dags trigger sp500_ml_pipeline_v3_production
```

**What the pipeline does:**
1. **Health Check** - Verifies environment (30 sec)
2. **Data Collection** - Market + News with validation (5 min)
3. **Feature Engineering** - Parallel processing (3 types simultaneously) (10 min)
4. **Gold Layer Merge** - Optimized merge (100x faster) (10 sec)
5. **News Processing** - Advanced sentiment analysis (5 min)
6. **Label Generation** - Multiple horizons (30-min, 60-min) (1 min)
7. **Model Training** - 5 models in parallel (10-15 min):
   - XGBoost Original (66 features)
   - **XGBoost Enhanced (114 features)** ⭐ Best: OOT AUC 0.5123
   - LightGBM Original (66 features)
   - XGBoost Regression
   - XGBoost 60-min horizon
8. **Model Selection** - Automated (OOT AUC ≥ 0.50, overfitting < 25%)
9. **Deployment** - Production deployment + monitoring
10. **Cleanup** - Remove temp files

**Expected Runtime:** ~45 minutes

### Step 5: Monitor Execution

**Watch logs in real-time:**
```bash
# All Airflow logs
docker-compose logs -f airflow-scheduler airflow-webserver

# Specific service
docker-compose logs -f mlflow
docker-compose logs -f fastapi
docker-compose logs -f streamlit
```

**In Airflow UI:**
- Click on DAG name to see graph view
- Green boxes = Success ✅
- Red boxes = Failed ❌
- Yellow boxes = Running 🏃
- Click any task → "Log" button to see detailed logs

### Step 6: View Results

**MLflow Experiments:**
```bash
# Open MLflow UI
open http://localhost:5001

# Navigate to:
# - Experiments → Compare 5 model variants
# - Models → View registered models
# - Artifacts → Download trained models
```

**Streamlit Dashboard:**
```bash
# Open interactive dashboard
open http://localhost:8501

# Features:
# - Real-time predictions
# - News snippets with sentiment
# - Price forecast with confidence bands
# - Sentiment timeline
# - Prediction history
# - Model metrics (ROC curve, confusion matrix)
# - Feature importance charts
```

---

## Component-Specific Demos

### Demo 1: Data Collection (Historical News)

Collect 5 years of free news data from GDELT:

```bash
# Activate virtual environment
source .venv/bin/activate

# Collect historical S&P 500 news (2020-2025)
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2020-10-19 \
    --end-date 2025-10-19 \
    --sources gdelt \
    --fetch-content \
    --max-workers 1 \
    --delay-between-requests 2.0
```

**What This Does:**
- Connects to GDELT Project API (free, unlimited)
- Filters for S&P 500-related articles
- Fetches full article content (handles 429 errors)
- Implements exponential backoff
- Downloads from 40+ news sources
- Auto-deduplicates and caches content
- Progress tracked in `seen_articles.json`

**Expected Output:**
- 50,000-100,000 articles with full text
- Cost: $0 (saves $999-$120,000/year vs paid APIs)
- Storage: `data_clean/bronze/news/hybrid/*.json`

### Demo 2: Feature Engineering

Process market and news features:

```bash
# Silver Layer: Technical indicators
python src_clean/data_pipelines/silver/market_technical_processor.py \
    --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
    --output data_clean/silver/market/technical/spx500_technical.csv

# Silver Layer: Microstructure metrics
python src_clean/data_pipelines/silver/market_microstructure_processor.py \
    --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
    --output data_clean/silver/market/microstructure/spx500_microstructure.csv

# Silver Layer: Volatility estimators
python src_clean/data_pipelines/silver/market_volatility_processor.py \
    --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
    --output data_clean/silver/market/volatility/spx500_volatility.csv

# Gold Layer: Merge market features (optimized!)
python src_clean/data_pipelines/gold/market_gold_builder.py \
    --technical data_clean/silver/market/technical/spx500_technical.csv \
    --microstructure data_clean/silver/market/microstructure/spx500_microstructure.csv \
    --volatility data_clean/silver/market/volatility/spx500_volatility.csv \
    --output data_clean/gold/market/features/spx500_features.csv

# Gold Layer: FinBERT trading signals
python src_clean/data_pipelines/gold/news_signal_builder.py \
    --silver-sentiment data_clean/silver/news/sentiment/spx500_sentiment.csv \
    --bronze-news data_clean/bronze/news \
    --output data_clean/gold/news/signals/spx500_news_signals.csv \
    --window 60
```

**Total Features Generated: 114**
- 64 market features (technical + microstructure + volatility)
- 6 FinBERT news signals
- 44 derived features (time-based, interactions)

### Demo 3: Model Training & Selection

Train multiple model variants:

```bash
# Train XGBoost Enhanced (114 features) - BEST MODEL
python src_clean/training/xgboost_training_pipeline_mlflow.py \
    --market-features data_clean/gold/market/features/spx500_features.csv \
    --news-signals data_clean/gold/news/signals/spx500_news_signals.csv \
    --prediction-horizon 30 \
    --experiment-name sp500_xgboost_enhanced

# View results in MLflow
mlflow ui --backend-store-uri file:./mlruns --port 5002
```

**Best Model Performance:**
- Train AUC: 0.5523
- Validation AUC: 0.5412
- Test AUC: 0.5089
- **OOT AUC: 0.5123** ✓ (meets 0.50 threshold)
- **Overfitting: 4.0%** ✓ (< 25% threshold)
- Accuracy: 51.23%

### Demo 4: Real-Time Inference

Start FastAPI backend and test endpoints:

```bash
# Terminal 1: Launch REST API + WebSocket server
uvicorn src_clean.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Test Endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Generate prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instrument": "SPX500_USD",
    "timestamp": "2025-10-26T10:30:00"
  }'

# Get prediction history
curl http://localhost:8000/predictions/history?limit=10

# Get recent news
curl http://localhost:8000/news/recent?limit=5
```

**Response Example:**
```json
{
  "timestamp": "2025-10-26T10:30:00",
  "instrument": "SPX500_USD",
  "prediction": "up",
  "probability": 0.5234,
  "confidence": 0.7891,
  "signal_strength": 0.2341,
  "features_used": 114,
  "model_version": "xgboost_enhanced_v2",
  "latency_ms": 45
}
```

---

## Troubleshooting

### Issue 1: Port 5000 Already in Use (macOS)

**Symptom:**
```
Error: bind: address already in use (port 5000)
```

**Cause:** macOS ControlCenter uses port 5000 for AirPlay Receiver

**Solution:**
```bash
# Use the fixed startup script (MLflow on port 5001)
./fix_port_and_start.sh
```

**Manual Fix:**
MLflow has been reconfigured to use port 5001 in `docker-compose.yml`. Access at: http://localhost:5001

### Issue 2: Airflow DAGs Not Showing

**Symptom:** DAG list is empty in Airflow UI

**Solution:**
```bash
# Restart scheduler and dag-processor
docker-compose restart airflow-scheduler airflow-dag-processor

# Wait 30 seconds
sleep 30

# Refresh browser at http://localhost:8080
```

### Issue 3: MLflow UI Not Loading

**Symptom:** Cannot access http://localhost:5001

**Solution:**
```bash
# Check MLflow status
docker-compose ps mlflow

# View logs
docker-compose logs mlflow

# Restart if needed
docker-compose restart mlflow

# Try accessing in browser (clear cache if needed)
open http://localhost:5001
```

### Issue 4: Docker Build Fails

**Symptom:** Build errors during `docker-compose build`

**Solution:**
```bash
# Clean build (no cache)
docker-compose build --no-cache airflow-init

# Remove old images
docker image prune -f

# Retry build
docker-compose build airflow-init
```

### Issue 5: Service Unhealthy

**Symptom:** `docker-compose ps` shows "unhealthy" status

**Solution:**
```bash
# Check specific service logs
docker-compose logs <service-name>

# Common fixes:
# - PostgreSQL: Wait longer (can take 30-60 seconds)
# - Redis: Restart: docker-compose restart redis
# - Airflow: Rebuild: docker-compose build airflow-init
```

### Issue 6: Prediction History Not Showing

**Symptom:** Dashboard shows no prediction history

**Solution:**
```bash
# Generate some predictions first
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instrument": "SPX500_USD"}'

# Check history file exists
ls -la data_clean/predictions/history.json

# Restart Streamlit
docker-compose restart streamlit
```

### Useful Commands

```bash
# View all service logs
docker-compose logs

# Follow logs for specific service
docker-compose logs -f airflow-scheduler

# Check service status
docker-compose ps

# Restart specific service
docker-compose restart <service-name>

# Stop all services
docker-compose down

# Stop and remove all data (clean slate)
docker-compose down -v

# Rebuild specific service
docker-compose build <service-name>

# View resource usage
docker stats

# Access service shell
docker-compose exec <service-name> bash
```

---

## Additional Resources

**Documentation:**
- [README.md](README.md) - Complete project overview
- [Airflow Setup Guide](docs/AIRFLOW_SETUP_GUIDE.md) - Detailed Airflow setup
- [DAG Testing Guide](docs/DAG_TESTING_GUIDE.md) - Testing procedures
- [Frontend Improvements](docs/FRONTEND_IMPROVEMENTS.md) - Dashboard v2.0 features
- [Airflow Fixes](docs/AIRFLOW_FIXES.md) - All applied fixes

**Quick Reference:**
- Local MLflow: http://localhost:5002
- Docker MLflow: http://localhost:5001
- Airflow UI: http://localhost:8080 (admin/admin)
- FastAPI Docs: http://localhost:8000/docs
- Streamlit: http://localhost:8501

---

**Version:** 2.0
**Last Updated:** November 1, 2025
**Status:** ✅ Fully Operational
