# S&P 500 ML Prediction Pipeline

End-to-end machine learning pipeline for S&P 500 price prediction using technical indicators and news sentiment.

## 🎯 Status

✅ **Production Ready** - Full pipeline operational with Python 3.11

## 🐳 Docker Quick Start

**New unified Docker structure** - All services in one `docker-compose.yml` at project root!

```bash
# Start entire MLOps stack
docker-compose up -d

# Access services:
# - Airflow: http://localhost:8080 (admin/admin)
# - MLflow: http://localhost:5000
# - FastAPI: http://localhost:8000/docs
# - Streamlit: http://localhost:8501
# - Evidently: http://localhost:8050

# View running services
docker-compose ps


```

**Services included:**
- **Orchestration**: Airflow 2.10.6 with custom task images (ETL, Trainer, DQ)
- **MLOps**: MLflow (tracking), Feast (features), Evidently (monitoring)
- **APIs**: FastAPI backend, Model servers (blue/green deployment)
- **UI**: Streamlit dashboard
- **Infrastructure**: PostgreSQL, Redis, Nginx load balancer

> **Note**: Clean Docker structure with organized subdirectories.

## 🚀 Quick Demo (5 minutes)

### Prerequisites
- Python 3.11
- OANDA practice account (get free API credentials at https://www.oanda.com/us-en/trading/api/)

### Setup
```bash
# 1. Clone and setup environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure OANDA credentials in .env
cat > .env << EOF
OANDA_TOKEN=your_token_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENV=practice
EOF
```

### Run the Pipeline
```bash
# (Optional) Terminal 1: Launch MLflow tracking UI on port 5002
source .venv/bin/activate
mlflow ui --backend-store-uri file:./mlruns --port 5002 --host 0.0.0.0
# Access: http://localhost:5002

# Terminal 2: Execute full medallion pipeline + model training
source .venv/bin/activate
python src_clean/run_full_pipeline.py \
  --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
  --bronze-news data_clean/bronze/news \
  --output-dir data_clean \
  --prediction-horizon 30

# Need features only? Skip news processing or training with flags:
# python src_clean/run_full_pipeline.py \
#   --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
#   --skip-news \
#   --skip-training

# Optional (after pipeline completes): Launch Streamlit dashboard
streamlit run src_clean/ui/streamlit_dashboard.py --server.headless true
```

### What You'll See
1. **Pipeline logs**: Stage-by-stage progress (Bronze→Silver→Gold→Training) with duration metrics
2. **ML artifacts** stored under `data_clean/`:
   - Gold features: `data_clean/gold/market/features/spx500_features.csv`
   - Gold labels: `data_clean/gold/market/labels/spx500_labels_30min.csv`
   - Trained models + metrics: `data_clean/models/xgboost_*`
3. **MLflow UI** (http://localhost:5002, if started): experiment runs, metrics, parameters, artifacts
4. **Streamlit Dashboard** (http://localhost:8501, if launched): visual inspection of features, metrics, and model outputs

## 📚 Complete System Demo

30-minute end-to-end demo includes:
- ✅ Data ingestion (Market + Hybrid news scraper)
- ✅ Feature engineering (Bronze → Silver → Gold)
- ✅ Model training with MLflow tracking
- ✅ Interactive Streamlit dashboard
- ✅ Airflow workflow orchestration
- ✅ Evidently AI model monitoring
- ✅ FastAPI REST API + WebSocket
- ✅ Full stack Docker deployment

## 🏗️ Architecture

**Medallion Data Pipeline**:
```
Hybrid News Sources → Bronze → Silver (TextBlob) → Gold (FinBERT) → Model → Inference → Monitoring
        ↓                ↓           ↓                    ↓            ↓        ↓           ↓
  data_clean/bronze   Raw storage  Preprocessing     Trading signals   API   Dashboard   Evidently
```

## 🎨 Components

### Data Layer
- **Bronze**: OANDA market data (1.7M 1-min candles) + RSS/GDELT news feeds (12,950+ articles)
- **Silver**: Engineered features (market: 64 features) + news sentiment (TextBlob: 6 features)
- **Gold**: Trading-ready signals (market features + FinBERT news signals + labels)

### Feature Engineering (70 Features)
**Market Features (64):**
- Price: OHLC, returns, log returns
- Technical Indicators: RSI, MACD, Bollinger Bands, Stochastic
- Moving Averages: SMA (5/10/20/50), EMA (5/10/20/50)
- Momentum: ADX, ATR, ROC, rate of change
- Volatility: Historical vol (20/50 periods), Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang estimators
- Volume: MA, ratio, z-score, velocity, acceleration
- Range: TR, ATR ratios, high-low spread
- Microstructure: Price impact, order flow imbalance, illiquidity metrics
- Advanced: VWAP, close/VWAP ratio, spread proxies

**News Features (6):**
- Average sentiment score
- Signal strength
- Article count (recent)
- Quality score
- News age
- Availability flag

### ML Models
- **Classification**: XGBoost binary classifier (Up/Down direction)
  - Performance: AUC 0.6349, Accuracy 58.85%
- **Regression**: XGBoost regressor (Percentage returns)
  - Performance: RMSE 0.15%, MAE 0.09%
  - Note: Predicts returns (not absolute price) to avoid naive persistence
- **Tracking**: MLflow experiment tracking & model registry

### Real-Time Prediction System
- **Event-Driven Architecture**: Automatically triggers predictions when news arrives
- **Live Market Data**: OANDA S&P 500 futures (SPX500_USD) - 24/5 trading
- **File System Watcher**: Monitors news directory using watchdog library
- **Feature Calculation**: Real-time computation of all 70 features from OANDA API
- **News Integration**: Displays triggering article headline, source, sentiment in dashboard

### Services

| Service | Port | Description | Access |
|---------|------|-------------|--------|
| **Streamlit** | 8501 | Interactive ML dashboard | http://localhost:8501 |
| **FastAPI** | 8000 | REST API + WebSocket | http://localhost:8000/docs |
| **MLflow** | 5000 | Experiment tracking | http://localhost:5000 |
| **Airflow** | 8080 | Workflow orchestration (Airflow 2.10.6) | http://localhost:8080 (admin/admin) |
| **Evidently** | 8050 | Model monitoring | http://localhost:8050 |
| **Model Servers** | 8001/8002 | Blue/Green deployments | http://localhost:8088 (via Nginx) |
| **Feast** | 6566 | Feature store API | http://localhost:6566 |

> **Docker**: All services available via unified `docker-compose.yml` 

**Note**: When running locally (non-Docker), MLflow uses port 5002 to avoid conflict with macOS AirPlay Receiver.

## 📰 Historical News Collection Demo (5+ Years for FREE)

**NEW**: Build a complete ML training dataset with 50,000-100,000 historical news articles at zero cost!

### Step 1: Collect Historical News (1-3 hours, one-time)
```bash
# Activate virtual environment
source .venv/bin/activate

# Collect 5 years of S&P 500 news from GDELT Project (2020-2025)
# Production scraper with 429 error handling and full content fetching
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2020-10-19 \
    --end-date 2025-10-19 \
    --sources gdelt \
    --fetch-content \
    --max-workers 1 \
    --delay-between-requests 2.0

# ⏱️ Takes 1-2 weeks with conservative settings (avoids 429 errors)
# 📊 Expected: 50,000-100,000 articles with full text
# 💰 Cost: $0
# 📁 Saved to: data_clean/bronze/news/hybrid/*.json
```

**What happens:**
- Connects to GDELT Project API (free, unlimited)
- Filters for S&P 500 relevant articles using expanded keywords
- Fetches full article content from original URLs with 429 error handling
- Implements exponential backoff and per-domain rate limiting
- Downloads from 40+ news sources (Yahoo Finance, Reuters, Bloomberg, etc.)
- Automatically deduplicates articles
- Caches content for 7 days to avoid re-fetching
- Tracks progress in `seen_articles.json` (can resume if interrupted)

### Step 2: Process Sentiment Features (2-5 minutes)
```bash
# Silver Layer: Quick sentiment analysis with TextBlob (preprocessing)
python src_clean/data_pipelines/silver/news_sentiment_processor.py \
    --input-dir data_clean/bronze/news \
    --output data_clean/silver/news/sentiment/spx500_sentiment.csv

# Automatically processes:
# - Original RSS articles (data_clean/bronze/news/*.json)
# - Hybrid scraper articles (data_clean/bronze/news/hybrid/*.json)
# - Merges everything into one dataset
```

**Silver Layer Features (TextBlob):**
- Polarity (-1 to +1): Overall positive/negative tone
- Subjectivity (0 to 1): Opinion vs fact
- Financial sentiment: Using finance-specific keywords
- Policy tone: Hawkish/dovish/neutral (for Fed news)
- Confidence score: Reliability of sentiment

### Step 3: Generate Trading Signals with FinBERT (NEW!)
```bash
# Gold Layer: Advanced sentiment analysis with FinBERT
# Transforms sentiment → trading signals using financial-domain AI
python src_clean/data_pipelines/gold/news_signal_builder.py \
    --silver-sentiment data_clean/silver/news/sentiment/spx500_sentiment.csv \
    --bronze-news data_clean/bronze/news \
    --output data_clean/gold/news/signals/spx500_news_signals.csv \
    --window 60

# ⏱️ Processing time: ~1-2 minutes per 1000 articles (CPU)
# 🤖 Model: ProsusAI/finbert (financial sentiment transformer)
# 📊 Output: Time-windowed trading signals (buy/sell/hold)
```

**Gold Layer Features (FinBERT):**
- `avg_sentiment`: Financial-domain sentiment score (-1 to +1)
- `signal_strength`: Confidence-weighted signal magnitude
- `trading_signal`: Buy (1), Sell (-1), or Hold (0)
- `article_count`: Number of articles in time window
- `quality_score`: Average confidence across articles
- `positive_prob`, `negative_prob`, `neutral_prob`: FinBERT class probabilities

**FinBERT Benefits:**
- ✅ **Financial domain expertise**: Trained specifically on financial texts
- ✅ **Context-aware**: Understands "hawkish" means bearish for stocks
- ✅ **High accuracy**: 78%+ confidence scores
- ✅ **Production-ready**: Batch processing with progress bars
- ✅ **CPU-friendly**: No GPU required (though faster with GPU)

### Step 4: Run Full Pipeline with Historical News
```bash
# Complete pipeline: Market + News → Features → Training
python src_clean/run_full_pipeline.py \
    --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
    --bronze-news data_clean/bronze/news \
    --output-dir data_clean

# Processes:
# ✅ Stage 1: Market → Silver (technical, microstructure, volatility)
# ✅ Stage 2: News → Silver (TextBlob sentiment)
# ✅ Stage 3: Market → Gold (merge features)
# ✅ Stage 4: News → Gold (FinBERT trading signals) ← NEW!
# ✅ Stage 5: Generate prediction labels
# ✅ Stage 6: XGBoost model training with news signals
```

**New Pipeline Architecture:**
```
Bronze News (12,950 articles)
    ↓
Silver Sentiment (TextBlob) - Fast preprocessing
    ↓
Gold Signals (FinBERT) - Financial-domain AI ← NEW!
    ↓
Training (XGBoost with news features)
```

### Optional: Enhance Coverage with Free API Keys

**Get more recent news and sentiment scores:**

```bash
# 1. Get free API keys (no credit card required):
#    - Alpha Vantage: https://www.alphavantage.co/support/#api-key (25 calls/day)
#    - Finnhub: https://finnhub.io/register (60 calls/min, 1-year history)

# 2. Add to .env file:
echo "ALPHAVANTAGE_KEY=your_key_here" >> .env
echo "FINNHUB_KEY=your_key_here" >> .env

# 3. Collect from all sources:
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2024-10-19 \
    --end-date 2025-10-19 \
    --sources all \
    --fetch-content \
    --max-workers 2 \
    --delay-between-requests 1.0

# Collects from:
# - GDELT Project (unlimited, free)
# - Alpha Vantage (with sentiment scores)
# - Finnhub (market-specific news)
```

### Daily Updates (Optional - Set and Forget)

**Keep your dataset fresh automatically:**

```bash
# Setup daily news collection via cron
crontab -e

# Add this line (runs at 1 AM daily):
0 1 * * * cd /path/to/fx-ml-pipeline && source .venv/bin/activate && python3 src_clean/data_pipelines/bronze/hybrid_news_scraper.py --mode incremental --sources all --fetch-content --max-workers 1 --delay-between-requests 2.0 >> logs/news_scraper.log 2>&1

# Collects 100-500 new articles daily for free
```

### Troubleshooting

**No articles collected?**
```bash
# Test GDELT API directly
curl "https://api.gdeltproject.org/api/v2/doc/doc?query=stock%20market&mode=artlist&maxrecords=5&format=json"

# Should return JSON with articles
```

**Want to see what was collected?**
```bash
# Count total articles
find data_clean/bronze/news/hybrid -name "*.json" -not -name "seen_articles.json" | wc -l

# View a sample article
cat data_clean/bronze/news/hybrid/*.json | head -1 | python3 -m json.tool
```

**Check processed sentiment:**
```bash
# View first 10 processed articles
head -10 data_clean/silver/news/sentiment/spx500_sentiment.csv | column -t -s,
```

### What You Get

**Free Historical Data:**
- 50,000-100,000 news articles (2017-2025)
- 40+ premium news sources (Yahoo Finance, Reuters, Bloomberg, CNBC, etc.)
- 12+ languages (automatically detected)
- S&P 500 filtered (only relevant market news)
- Saves $999-$120,000/year vs paid alternatives

**Compatible Sources:**
- ✅ GDELT Project (2017-present, unlimited, FREE)
- ✅ Alpha Vantage (25 calls/day, FREE, includes sentiment scores)
- ✅ Finnhub (60 calls/min, 1-year history, FREE)
- ✅ Original RSS feeds (real-time, FREE)
- ✅ All sources auto-merge and deduplicate

**Ready for ML Training:**
- Integrates seamlessly with existing pipeline
- Bronze → Silver → Gold architecture maintained
- Combines with market data automatically
- No code changes needed

---

## 🔄 Demo Workflows

### 1. Data Ingestion Demo
```bash
# Collect recent S&P 500 news (incremental run with content fetching)
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
  --mode recent \
  --sources gdelt \
  --fetch-content \
  --max-workers 1 \
  --delay-between-requests 2.0

# (Optional) Backfill a specific date range
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
  --start-date 2025-01-01 \
  --end-date 2025-10-26 \
  --sources all \
  --fetch-content \
  --max-workers 1 \
  --delay-between-requests 2.0

# Generate Silver-layer sentiment features
python src_clean/data_pipelines/silver/news_sentiment_processor.py \
  --input-dir data_clean/bronze/news \
  --output data_clean/silver/news/sentiment/spx500_sentiment.csv

# Build Gold-layer FinBERT trading signals
python src_clean/data_pipelines/gold/news_signal_builder.py \
  --silver-sentiment data_clean/silver/news/sentiment/spx500_sentiment.csv \
  --bronze-news data_clean/bronze/news \
  --output data_clean/gold/news/signals/spx500_news_signals.csv
```

### 2. Training Demo
```bash
# Train with MLflow tracking
python src_clean/training/xgboost_training_pipeline_mlflow.py \
  --market-features data_clean/gold/market/features/spx500_features.csv \
  --news-signals data_clean/gold/news/signals/spx500_news_signals.csv \
  --prediction-horizon 30 \
  --experiment-name demo_experiment

# View results: http://localhost:5002
```

### 3. Inference Demo
```bash
# Start FastAPI
uvicorn src_clean.api.main:app --host 0.0.0.0 --port 8000 &

# Generate prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instrument": "SPX500_USD"}'

# Stream real-time updates (requires wscat)
wscat -c ws://localhost:8000/ws/market-stream
```

### 4. Orchestration Demo
```bash
# Start Airflow (version 2.10.6)
cd docker/airflow
docker compose up -d airflow-scheduler airflow-webserver airflow-triggerer postgres redis

# Access: http://localhost:8080
# Login: admin / admin
# Trigger DAGs:
#   - data_pipeline (Bronze → Silver → Gold)
#   - train_deploy_pipeline (Train & deploy model)
#   - batch_inference (Generate predictions)
```

### 5. Monitoring Demo
```bash
# Start Evidently
docker-compose up -d evidently-monitor

# Generate drift report
curl -X POST http://localhost:8050/generate

# View: http://localhost:8050/latest_report.html
```

### 6. Full Stack Demo
```bash
# Launch all services (unified docker-compose at root)
docker-compose up -d

# Or start specific service groups:
# MLOps stack only
docker-compose up -d postgres redis mlflow feast

# Airflow orchestration
docker-compose up -d airflow-postgres airflow-webserver airflow-scheduler

# API & UI
docker-compose up -d fastapi streamlit

# Model servers with load balancer
docker-compose up -d model-blue model-green nginx

# Verify all services healthy
docker-compose ps

# Access all UIs (see Services table above)
```

## 📊 Features

**Market Features (37)**:
- Technical: RSI, MACD, Bollinger Bands, Moving Averages, ATR, ADX
- Microstructure: Volume patterns, spread proxies, order flow
- Volatility: Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang

**News Features (11)**:
- Sentiment scores (positive/negative/neutral)
- Trading signal strength
- Article quality metrics
- Policy tone indicators

## 📁 Project Structure

```
fx-ml-pipeline/
├── README.md                    # This file
├── requirements.txt             # Python 3.11 dependencies
├── pyproject.toml               # Poetry-style project metadata
│
├── src_clean/                   # Production code (Bronze → Silver → Gold → Training)
│   ├── api/                     # FastAPI backend
│   ├── ui/                      # Streamlit dashboards
│   ├── data_pipelines/          # Data ingestion & feature engineering
│   ├── training/                # Model training pipelines
│   └── utils/                   # Shared helpers
│
├── docker-compose.yml           # Unified Docker orchestration (root)
├── docker/                      # Docker build contexts
│   ├── airflow/                 # Airflow deployable image
│   ├── api/                     # FastAPI Dockerfile
│   ├── monitoring/              # Evidently monitoring containers
│   ├── tasks/                   # Airflow task images (ETL, trainer, DQ, model-server)
│   ├── tools/                   # Local developer utilities
│   └── ui/                      # Streamlit Dockerfile
│
├── configs/                     # Configuration files (YAML, JSON)
├── data_clean/                  # Medallion data outputs (bronze/silver/gold/models)
├── data_clean_5year/            # Sample 5-year datasets & trained models
├── scripts/                     # Utility shell scripts
├── tests/                       # Unit and integration tests
├── logs/                        # Local run logs (gitignored)
├── mlruns/                      # MLflow experiment store
├── outputs/                     # Generated reports/plots (gitignored)
└── archive/                     # Archived legacy code & assets
```

## 🔧 Requirements

- **Python**: 3.11+ (required)
- **Docker**: For full stack deployment
- **RAM**: 16GB minimum
- **Disk**: 50GB free space
- **OANDA Account**: Free demo account available

## 🎓 Learning Resources



## 🐛 Troubleshooting

### Python Version
```bash
# Verify Python 3.11
python --version  # Should show 3.11.x

# Recreate venv if needed
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Port Conflicts
```bash
# Find process
lsof -i :8000  # or any port

# Kill process
kill -9 <PID>
```

**macOS Port 5000 Conflict**: On macOS, AirPlay Receiver uses port 5000. This project uses **port 5002 for MLflow** to avoid conflicts.

To disable AirPlay Receiver (optional):
1. Open System Settings → General → AirDrop & Handoff
2. Turn off "AirPlay Receiver"

### Docker Issues
```bash
# Restart containers
docker-compose restart

# View logs
docker-compose logs -f <service>

# Rebuild specific service
docker-compose build <service>

# Start fresh (removes volumes)
docker-compose down -v
docker-compose up -d

# See detailed structure: DOCKER_STRUCTURE.md
```

## 📝 License

Educational and research purposes only.

---

**Version**: 2.0.0
**Python**: 3.11+
**Last Updated**: October 2025
