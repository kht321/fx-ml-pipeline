# FX ML Pipeline - Complete Demo Guide

## Table of Contents
- [Quick Start (5 Minutes)](#quick-start-5-minutes)
- [Full System Demo (30 Minutes)](#full-system-demo-30-minutes)
- [Docker Stack Demo](#docker-stack-demo)
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

**Option B: Streamlit Dashboard**
```bash
# Terminal 2: Launch interactive dashboard
streamlit run src_clean/ui/streamlit_dashboard.py

# Access at: http://localhost:8501
```

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

## Full System Demo (30 Minutes)

This comprehensive demo showcases all components of the production MLOps pipeline.

### Part 1: Historical Data Collection (10 minutes)

#### Collect 5 Years of Free News Data
```bash
# Activate virtual environment
source .venv/bin/activate

# Collect historical S&P 500 news from GDELT (2020-2025)
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2020-10-19 \
    --end-date 2025-10-19 \
    --sources gdelt \
    --fetch-content \
    --max-workers 1 \
    --delay-between-requests 2.0
```

**What This Does:**
- Connects to GDELT Project API (free, unlimited access)
- Filters for S&P 500-related articles using financial keywords
- Fetches full article content with 429 error handling
- Implements exponential backoff and per-domain rate limiting
- Downloads from 40+ news sources (Yahoo Finance, Reuters, Bloomberg)
- Auto-deduplicates and caches content
- Progress tracked in `seen_articles.json`

**Expected Output:**
- 50,000-100,000 articles with full text
- Cost: $0 (saves $999-$120,000/year vs paid alternatives)
- Storage: `data_clean/bronze/news/hybrid/*.json`

#### Process News Features
```bash
# Silver Layer: TextBlob sentiment analysis
python src_clean/data_pipelines/silver/news_sentiment_processor.py \
    --input-dir data_clean/bronze/news \
    --output data_clean/silver/news/sentiment/spx500_sentiment.csv

# Gold Layer: FinBERT trading signals
python src_clean/data_pipelines/gold/news_signal_builder.py \
    --silver-sentiment data_clean/silver/news/sentiment/spx500_sentiment.csv \
    --bronze-news data_clean/bronze/news \
    --output data_clean/gold/news/signals/spx500_news_signals.csv \
    --window 60
```

**FinBERT Features Generated:**
- `avg_sentiment`: Financial-domain sentiment (-1 to +1)
- `signal_strength`: Confidence-weighted magnitude
- `trading_signal`: Buy (1), Sell (-1), Hold (0)
- `article_count`: Articles in 60-min window
- `quality_score`: Average confidence
- `positive_prob`, `negative_prob`, `neutral_prob`

### Part 2: Feature Engineering Pipeline (5 minutes)

```bash
# Run complete feature engineering pipeline
python src_clean/run_full_pipeline.py \
    --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
    --bronze-news data_clean/bronze/news \
    --output-dir data_clean \
    --prediction-horizon 30
```

**Pipeline Stages:**
1. **Bronze → Silver (Market Technical)**: RSI, MACD, Bollinger Bands, Moving Averages (2-3 min)
2. **Bronze → Silver (Microstructure)**: Liquidity, spreads, order flow (1-2 min)
3. **Bronze → Silver (Volatility)**: 7 estimators including Garman-Klass, Yang-Zhang (2-3 min)
4. **Silver → Gold (Market Merge)**: Combine 64 base features (10 seconds)
5. **Bronze → Silver (News Sentiment)**: TextBlob analysis (5 min)
6. **Silver → Gold (FinBERT Signals)**: Transform to trading signals (1-2 min/1000 articles)
7. **Label Generation**: 30-min and 60-min price direction (1 min)

**Total Features Generated: 114**
- 64 market features (technical + microstructure + volatility)
- 6 FinBERT news signals
- 44 derived features (time-based, interactions, etc.)

### Part 3: Model Training & Selection (5 minutes)

```bash
# Train multiple model variants with MLflow tracking
python src_clean/training/xgboost_training_pipeline_mlflow.py \
    --market-features data_clean/gold/market/features/spx500_features.csv \
    --news-signals data_clean/gold/news/signals/spx500_news_signals.csv \
    --prediction-horizon 30 \
    --experiment-name sp500_xgboost_enhanced
```

**Models Trained:**
1. **XGBoost Classification (64 features)**: Base market features only
2. **XGBoost Classification (114 features)**: Enhanced with news signals ← **BEST**
3. **XGBoost Regression**: Percentage returns prediction
4. **LightGBM Classification**: Faster alternative
5. **XGBoost 60-min**: Longer prediction horizon

**Best Model Performance (XGBoost Enhanced, 114 features):**
- Train AUC: 0.5523
- Validation AUC: 0.5412
- Test AUC: 0.5089
- **OOT AUC: 0.5123** ✓ (meets 0.50 threshold)
- **Overfitting: 4.0%** ✓ (< 25% threshold)
- Accuracy: 51.23%

**Why These Metrics Matter:**
- AUC > 0.5 indicates better than random chance
- OOT (Out-of-Time) test on most recent 10% of data shows true future performance
- Low overfitting (4%) demonstrates good generalization
- Even 51% accuracy is valuable in financial markets

### Part 4: MLflow Experiment Tracking (2 minutes)

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5002 --host 0.0.0.0
```

**Access:** http://localhost:5002

**Features to Explore:**
- **Experiments**: Compare 5 model variants side-by-side
- **Metrics**: AUC, accuracy, precision, recall, F1 score
- **Parameters**: Hyperparameters (max_depth, learning_rate, etc.)
- **Artifacts**: Trained models (.pkl), plots (ROC curves, feature importance)
- **Model Registry**: Version models, promote to staging/production

**Key Comparisons:**
```
Model                       | OOT AUC | Features | Overfitting
----------------------------|---------|----------|-------------
XGBoost Enhanced (Best)     | 0.5123  | 114      | 4.0%
XGBoost Original            | 0.5089  | 64       | 5.2%
LightGBM Original           | 0.5067  | 64       | 6.1%
XGBoost Regression          | N/A     | 64       | RMSE 0.15%
XGBoost 60-min              | 0.5045  | 114      | 7.3%
```

### Part 5: Real-Time Inference (5 minutes)

#### Start FastAPI Backend
```bash
# Terminal 1: Launch REST API + WebSocket server
uvicorn src_clean.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Access API Docs:** http://localhost:8000/docs

#### Test Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-26T10:00:00",
  "model_loaded": true,
  "feast_available": true,
  "redis_connected": true
}
```

**Generate Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instrument": "SPX500_USD",
    "timestamp": "2025-10-26T10:30:00"
  }'
```

**Response:**
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

**WebSocket Streaming (requires wscat):**
```bash
# Install wscat if needed: npm install -g wscat
wscat -c ws://localhost:8000/ws/market-stream
```

**Stream Output (every 5 seconds):**
```json
{
  "type": "market_update",
  "timestamp": "2025-10-26T10:30:00",
  "data": {
    "instrument": "SPX500_USD",
    "price": 4521.50,
    "volume": 1234567,
    "prediction": "up",
    "probability": 0.5234,
    "confidence": 0.7891
  }
}
```

### Part 6: Interactive Dashboard (3 minutes)

```bash
# Terminal 2: Launch Streamlit UI
streamlit run src_clean/ui/streamlit_dashboard.py
```

**Access:** http://localhost:8501

**Dashboard Features:**
1. **Real-time Price & Prediction**
   - SPX500 current price from OANDA API
   - ML prediction (up/down/neutral) with confidence
   - Live updates every 5 seconds

2. **Feature Importance**
   - Top-20 features driving predictions
   - Interactive bar charts
   - Feature correlation heatmap

3. **Model Metrics**
   - AUC-ROC curve
   - Confusion matrix
   - Precision/recall tradeoff
   - Calibration curve

4. **News Sentiment**
   - Recent articles with sentiment scores
   - FinBERT signal strength indicators
   - Signal distribution (buy/sell/hold)

5. **Historical Performance**
   - Prediction accuracy over time
   - Win rate by hour/day of week
   - Performance analytics

6. **System Health**
   - Data pipeline status
   - Model freshness
   - API latency
   - Service availability

---

## Docker Stack Demo

Deploy the complete production MLOps stack with one command.

### Full Stack Deployment

```bash
# Start all 16 services
docker-compose up -d

# Verify all services healthy
docker-compose ps
```

**Services Started:**
```
NAME                    STATUS      PORTS
postgres                Up          5432
redis                   Up          6379
mlflow                  Up          5000
feast                   Up          6566
airflow-postgres        Up          5433
airflow-webserver       Up          8080
airflow-scheduler       Up          -
airflow-init            Exit 0      -
fastapi                 Up          8000
streamlit               Up          8501
model-blue              Up          8001
model-green             Up          8002
nginx                   Up          8088
evidently               Up          8050
```

### Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow** | http://localhost:8080 | admin / admin |
| **MLflow** | http://localhost:5000 | None |
| **FastAPI** | http://localhost:8000/docs | None |
| **Streamlit** | http://localhost:8501 | None |
| **Evidently** | http://localhost:8050 | None |
| **Model API** | http://localhost:8088/predict | None (via Nginx) |
| **Feast** | http://localhost:6566 | None |

### Trigger Airflow Pipeline

```bash
# Access Airflow UI: http://localhost:8080
# Login: admin / admin
# Enable DAG: sp500_ml_pipeline_v3_production
# Click "Trigger DAG" button

# Or trigger via API:
curl -X POST http://localhost:8080/api/v1/dags/sp500_ml_pipeline_v3_production/dagRuns \
  -H "Content-Type: application/json" \
  -u "admin:admin" \
  -d '{}'
```

**DAG Stages (9 steps):**
1. Data Collection (OANDA + News)
2. Feature Engineering (Bronze → Silver → Gold)
3. News Processing (FinBERT)
4. Label Generation (30-min, 60-min)
5. Model Training (5 variants)
6. Model Selection (Best OOT AUC)
7. Deployment (Copy to production/)
8. Monitoring (Evidently reports)
9. Cleanup (Old data)

**Expected Duration:** 30-60 minutes

### Blue/Green Deployment Demo

```bash
# Check current model serving
curl http://localhost:8088/health

# Response shows active slot:
# {"status": "ok", "slot": "blue"}

# Deploy new model to green slot
docker-compose restart model-green

# Update Nginx config to switch traffic (90% blue, 10% green for canary)
# Edit docker/load-balancer/nginx.conf
# Reload Nginx:
docker-compose exec nginx nginx -s reload

# Full cutover: Update to 100% green, restart Nginx
docker-compose restart nginx
```

### Monitoring with Evidently

```bash
# Generate drift report
curl -X POST http://localhost:8050/generate

# View report
open http://localhost:8050/latest_report.html
```

**Report Contents:**
- Data drift detection (feature distribution changes)
- Model performance degradation
- Prediction drift
- Feature statistics

### Service-Specific Demos

**MLOps Stack Only:**
```bash
docker-compose up -d postgres redis mlflow feast
```

**Airflow Only:**
```bash
docker-compose up -d airflow-postgres airflow-webserver airflow-scheduler
```

**API & UI Only:**
```bash
docker-compose up -d fastapi streamlit
```

**Model Servers with Load Balancer:**
```bash
docker-compose up -d model-blue model-green nginx
```

### Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (caution: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

---

## Component-Specific Demos

### Data Pipeline Components

#### 1. Market Technical Features
```bash
python src_clean/data_pipelines/silver/market_technical_processor.py \
    --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
    --output data_clean/silver/market/technical/spx500_technical.csv
```

**Features Created (17):**
- RSI (14-period)
- MACD (12, 26, 9)
- Bollinger Bands (20-period, 2 std)
- Stochastic Oscillator
- SMA (5, 10, 20, 50 periods)
- EMA (5, 10, 20, 50 periods)
- ATR (14-period)
- ADX (14-period)

#### 2. Market Microstructure Features
```bash
python src_clean/data_pipelines/silver/market_microstructure_processor.py \
    --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
    --output data_clean/silver/market/microstructure/spx500_microstructure.csv
```

**Features Created (7):**
- Bid/ask liquidity
- Effective spread
- Quoted depth
- Order flow imbalance
- Price impact
- Liquidity shocks
- Illiquidity ratio

#### 3. Volatility Features
```bash
python src_clean/data_pipelines/silver/market_volatility_processor.py \
    --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
    --output data_clean/silver/market/volatility/spx500_volatility.csv
```

**Features Created (7):**
- Historical volatility (20, 50 periods)
- Garman-Klass estimator
- Parkinson estimator
- Rogers-Satchell estimator
- Yang-Zhang estimator
- Range-based volatility
- Volatility percentile rank

#### 4. News Sentiment Processing
```bash
# Silver Layer: TextBlob sentiment
python src_clean/data_pipelines/silver/news_sentiment_processor.py \
    --input-dir data_clean/bronze/news \
    --output data_clean/silver/news/sentiment/spx500_sentiment.csv

# Gold Layer: FinBERT signals
python src_clean/data_pipelines/gold/news_signal_builder.py \
    --silver-sentiment data_clean/silver/news/sentiment/spx500_sentiment.csv \
    --bronze-news data_clean/bronze/news \
    --output data_clean/gold/news/signals/spx500_news_signals.csv
```

**TextBlob Features (5):**
- Polarity (-1 to +1)
- Subjectivity (0 to 1)
- Financial sentiment score
- Policy tone (hawkish/dovish/neutral)
- Confidence score

**FinBERT Features (6):**
- Average sentiment
- Signal strength
- Trading signal (buy/sell/hold)
- Article count
- Quality score
- Class probabilities (positive/negative/neutral)

### Model Training Variants

#### XGBoost Classification (Enhanced)
```bash
python src_clean/training/xgboost_training_pipeline_mlflow.py \
    --market-features data_clean/gold/market/features/spx500_features.csv \
    --news-signals data_clean/gold/news/signals/spx500_news_signals.csv \
    --prediction-horizon 30 \
    --experiment-name sp500_xgboost_enhanced
```

#### XGBoost Regression
```bash
python src_clean/training/xgboost_training_pipeline_mlflow.py \
    --market-features data_clean/gold/market/features/spx500_features.csv \
    --prediction-horizon 30 \
    --task-type regression \
    --experiment-name sp500_xgboost_regression
```

#### LightGBM Classification
```bash
python src_clean/training/lightgbm_training_pipeline.py \
    --market-features data_clean/gold/market/features/spx500_features.csv \
    --prediction-horizon 30 \
    --experiment-name sp500_lightgbm
```

#### Model Selection
```bash
# Compare all experiments and select best model
python src_clean/training/multi_experiment_selector.py \
    --experiments sp500_xgboost_enhanced sp500_xgboost_original sp500_lightgbm \
    --output data_clean/models/best_model_selection.json
```

### Feature Store (Feast) Demo

```bash
# Apply feature definitions
cd feature_repo
feast apply

# Materialize features to online store (Redis)
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

# Get online features
python - <<EOF
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")

# Get latest features for SPX500_USD
entity_rows = pd.DataFrame({
    "instrument_id": ["SPX500_USD"],
    "event_timestamp": [pd.Timestamp.now()]
})

features = store.get_online_features(
    features=[
        "market_features:rsi",
        "market_features:macd",
        "news_features:avg_sentiment"
    ],
    entity_rows=entity_rows
).to_dict()

print(features)
EOF
```

---

## Troubleshooting

### Common Issues

#### 1. Python Version Mismatch
```bash
# Check Python version
python --version  # Should be 3.11.x

# If wrong version, recreate venv
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. OANDA API Errors
```bash
# Test OANDA connection
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "https://api-fxpractice.oanda.com/v3/accounts/YOUR_ACCOUNT_ID"

# Should return account details in JSON
```

#### 3. Port Conflicts

**macOS Port 5000 Issue:**
- AirPlay Receiver uses port 5000
- This project uses **port 5002 for MLflow** locally
- Docker uses port 5000 (no conflict as it's isolated)

**Disable AirPlay Receiver (optional):**
1. System Settings → General → AirDrop & Handoff
2. Turn off "AirPlay Receiver"

**Find and kill process:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

#### 4. Docker Issues

**Services not starting:**
```bash
# Check service logs
docker-compose logs -f <service-name>

# Restart specific service
docker-compose restart <service-name>

# Rebuild image
docker-compose build <service-name>
docker-compose up -d <service-name>
```

**Clean restart:**
```bash
# Stop all services
docker-compose down

# Remove volumes (caution: deletes data)
docker-compose down -v

# Start fresh
docker-compose up -d
```

#### 5. Memory Issues

**Pipeline running out of memory:**
```bash
# Reduce batch size in processing scripts
# Add --batch-size parameter

python src_clean/data_pipelines/silver/market_technical_processor.py \
    --input data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
    --output data_clean/silver/market/technical/spx500_technical.csv \
    --batch-size 100000  # Smaller batches
```

**Docker memory limits:**
```bash
# Check Docker resource usage
docker stats

# Increase Docker memory limit:
# Docker Desktop → Settings → Resources → Memory
# Set to at least 8GB
```

#### 6. News Scraper Issues

**No articles collected:**
```bash
# Test GDELT API
curl "https://api.gdeltproject.org/api/v2/doc/doc?query=stock%20market&mode=artlist&maxrecords=5&format=json"

# Should return JSON with articles
```

**429 Errors (Too Many Requests):**
```bash
# Increase delay between requests
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2024-10-19 \
    --end-date 2025-10-19 \
    --sources gdelt \
    --delay-between-requests 5.0  # Increased from 2.0
```

#### 7. MLflow Tracking Issues

**Cannot connect to MLflow:**
```bash
# Check if server is running
curl http://localhost:5002

# Start MLflow server
mlflow ui --backend-store-uri file:./mlruns --port 5002
```

**Experiments not showing:**
```bash
# List experiments
mlflow experiments list

# Search runs
mlflow runs list --experiment-name sp500_xgboost_enhanced
```

#### 8. Feast Issues

**Features not available:**
```bash
# Check Feast registry
feast registry-dump

# Re-apply features
cd feature_repo
feast apply

# Materialize to online store
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### Performance Optimization

#### Slow Feature Engineering
```bash
# Use multiprocessing
export NUM_WORKERS=4

# Profile code
python -m cProfile -o profile.stats src_clean/run_full_pipeline.py ...
python -m pstats profile.stats
```

#### Slow Training
```bash
# Use GPU acceleration (if available)
# Add to training script:
# tree_method='gpu_hist'

# Reduce features for quick iterations
# Use --feature-subset parameter

# Sample data
# Add --sample-ratio 0.1 to use 10% of data
```

#### Slow Inference
```bash
# Cache features in Redis
# Feast automatically caches online features

# Reduce model complexity
# Train with smaller max_depth

# Use LightGBM instead of XGBoost
# Generally 2-3x faster
```

---

## Additional Resources

### Documentation
- [Complete Architecture Documentation](docs/COMPLETE_ARCHITECTURE_DOCUMENTATION.md)
- [Quick Reference Guide](docs/QUICK_REFERENCE.md)
- [Exploration Summary](docs/EXPLORATION_SUMMARY.txt)
- [Docker Structure](docs/DOCKER_STRUCTURE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

### External Links
- [OANDA API Documentation](https://developer.oanda.com)
- [GDELT Project](https://www.gdeltproject.org/)
- [FinBERT Model](https://huggingface.co/ProsusAI/finbert)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Feast Documentation](https://docs.feast.dev/)

### Support
For issues or questions:
1. Check troubleshooting section above
2. Review logs in `logs/` directory
3. Check Docker service logs: `docker-compose logs -f <service>`
4. Review documentation in `docs/` directory

---

**Last Updated:** October 26, 2025
**Version:** 2.0 (Production Ready)
**Python:** 3.11+
**Docker:** Compose v2.0+
