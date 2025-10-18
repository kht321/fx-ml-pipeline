# S&P 500 ML Prediction Pipeline

End-to-end machine learning pipeline for S&P 500 price prediction using technical indicators and news sentiment.

## 🎯 Status

✅ **Production Ready** - Full pipeline operational with Python 3.11

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

### Run the Demo
```bash
# Terminal 1: Start News Simulator (port 5001)
cd news-simulator
python app.py
# Keep running - generates test news articles

# Terminal 2: Start MLflow Tracking (port 5002, avoids macOS AirPlay conflict)
cd ..
source .venv/bin/activate
mlflow ui --backend-store-uri file:./mlruns --port 5002 --host 0.0.0.0
# Access: http://localhost:5002

# Terminal 3: Train Model (one-time setup)
source .venv/bin/activate
python src_clean/training/xgboost_training_pipeline_mlflow.py \
  --market-features data_clean/gold/market/features/spx500_features.csv \
  --news-signals data_clean/gold/news/signals/sp500_trading_signals.csv \
  --prediction-horizon 30 \
  --mlflow-uri http://localhost:5002
# Wait for training to complete (~2-3 minutes)

# Terminal 4: Start Event-Driven Predictor
source .venv/bin/activate
python src_clean/ui/realtime_predictor.py
# Keep running - auto-generates predictions when news arrives
# Uses OANDA API for real-time S&P 500 futures data (24/5 trading)

# Terminal 5: Launch Streamlit Dashboard (port 8501)
source .venv/bin/activate
streamlit run src_clean/ui/streamlit_dashboard.py
# Access: http://localhost:8501

# Terminal 6: Trigger Predictions
# Simulate positive news
curl -X POST http://localhost:5001/api/stream/positive

# Simulate negative news
curl -X POST http://localhost:5001/api/stream/negative

# Watch Terminal 4 - you'll see:
# "INFO - Fetched 200 candles from OANDA for SPX500_USD"
# "INFO - Prediction: UP/DOWN (confidence: XX.XX%)"

# Refresh Streamlit dashboard (Tab 2) to see:
# - Latest prediction with confidence
# - News headline that triggered it
# - All 70 features calculated from real OANDA data
# - Sentiment analysis of the news
```

### What You'll See
1. **MLflow UI** (http://localhost:5002): Track experiments, view metrics, compare models
2. **Streamlit Dashboard** (http://localhost:8501):
   - **Tab 1**: Live S&P 500 price charts with candlesticks
   - **Tab 2**: Event-driven predictions with news headlines
   - **Tab 3**: Model performance metrics (AUC, accuracy, confusion matrix)
   - **Tab 4**: Feature importance analysis (top 20 features)
3. **Real-time Predictions**: Automatically triggered when news arrives, using:
   - ✅ Live OANDA S&P 500 futures data (SPX500_USD)
   - ✅ 70 calculated features (64 market + 6 news)
   - ✅ News sentiment analysis
   - ✅ XGBoost model inference

## 📚 Complete System Demo

**For comprehensive walkthrough**, see **[Complete Demo Guide](docs/COMPLETE_DEMO_GUIDE.md)**

30-minute demo includes:
- ✅ Data ingestion (Market + News simulator)
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
News Simulator → Bronze → Silver → Gold → Model → Inference → Monitoring
      ↓            ↓        ↓        ↓      ↓        ↓           ↓
   5001 port    Raw    Features  Training  API   Dashboard   Evidently
              Storage  Engineering         8000    8501        8050
```

## 🎨 Components

### Data Layer
- **Bronze**: OANDA market data (1.7M 1-min candles) + RSS news feeds
- **Silver**: Engineered features (market + news sentiment)
- **Gold**: Training-ready dataset with labels

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
| **MLflow** | 5002 | Experiment tracking | http://localhost:5002 |
| **Airflow** | 8080 | Workflow orchestration (Airflow 2.10.3) | http://localhost:8080 (admin/admin) |
| **Evidently** | 8050 | Model monitoring | http://localhost:8050 |
| **News Simulator** | 5001 | Test data generator | http://localhost:5001 |

**Note**: MLflow uses port 5002 (not 5000) to avoid conflict with macOS AirPlay Receiver.

## 📰 Historical News Collection (FREE)

**NEW**: Collect 5+ years of S&P 500 news for FREE using the hybrid scraper!

### Quick Start
```bash
# Activate virtual environment
source .venv/bin/activate

# Collect 2017-2025 historical news (~50k-100k articles in 1-3 hours)
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2017-01-01 \
    --end-date 2025-10-19 \
    --sources gdelt

# Cost: $0 | Articles: 50,000-100,000 | Time: 1-3 hours
```

### Features
- ✅ **GDELT Project**: 2017-present, unlimited free access
- ✅ **Alpha Vantage**: 25 calls/day (FREE with sentiment scores)
- ✅ **Finnhub**: 60 calls/min, 1 year history (FREE)
- ✅ **Automatic deduplication** across all sources
- ✅ **S&P 500 filtering** - only relevant articles
- ✅ **Compatible** with existing pipeline (Bronze/Silver/Gold)

### Optional: Add Free API Keys for More Coverage
```bash
# Get free keys:
# - Alpha Vantage: https://www.alphavantage.co/support/#api-key
# - Finnhub: https://finnhub.io/register

# Add to .env
echo "ALPHAVANTAGE_KEY=your_key" >> .env
echo "FINNHUB_KEY=your_key" >> .env

# Run with all sources
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2017-01-01 \
    --sources all
```

### Documentation
- **Quick Reference**: [docs/README_HYBRID_NEWS_SCRAPER.md](docs/README_HYBRID_NEWS_SCRAPER.md)
- **Complete Guide**: [docs/HYBRID_NEWS_SCRAPER_GUIDE.md](docs/HYBRID_NEWS_SCRAPER_GUIDE.md)
- **Configuration**: [configs/hybrid_news_sources.yaml](configs/hybrid_news_sources.yaml)

**Savings**: $999-$120,000/year vs paid alternatives!

---

## 🔄 Demo Workflows

### 1. Data Ingestion Demo
```bash
# Start news simulator
cd news-simulator && python app.py &

# Stream 100 test articles (40 positive, 30 neutral, 30 negative)
for i in {1..40}; do curl -X POST http://localhost:5001/api/stream/positive; done
for i in {1..30}; do curl -X POST http://localhost:5001/api/stream/neutral; done
for i in {1..30}; do curl -X POST http://localhost:5001/api/stream/negative; done

# Articles automatically saved to data/news/bronze/simulated/
# Copy to processing directory
cp data/news/bronze/simulated/*.json data_clean/bronze/news/raw_articles/

# Run sentiment analysis (Silver layer)
python src_clean/data_pipelines/silver/news_sentiment_processor.py \
  --input data_clean/bronze/news/raw_articles/ \
  --output data_clean/silver/news/sentiment/
```

### 2. Training Demo
```bash
# Train with MLflow tracking
python src_clean/training/xgboost_training_pipeline_mlflow.py \
  --market-features data_clean/gold/market/features/spx500_features.csv \
  --news-signals data_clean/gold/news/signals/sp500_trading_signals.csv \
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
# Start Airflow (version 2.10.3)
cd airflow_mlops
docker compose up -d postgres-airflow airflow-web airflow-scheduler

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
# Launch all services
cd docker
docker compose -f docker-compose.full-stack.yml up -d

# Verify all services healthy
docker compose -f docker-compose.full-stack.yml ps

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
│
├── src_clean/                   # Production code
│   ├── api/                     # FastAPI backend
│   ├── ui/                      # Streamlit dashboards
│   ├── data_pipelines/          # Bronze → Silver → Gold
│   │   ├── bronze/              # Data collection
│   │   ├── silver/              # Feature engineering
│   │   └── gold/                # Training data prep
│   ├── training/                # XGBoost training
│   │   ├── xgboost_training_pipeline.py
│   │   └── xgboost_training_pipeline_mlflow.py
│   └── utils/                   # Shared utilities
│
├── docker/                      # Docker configurations
│   ├── Dockerfile.fastapi
│   ├── Dockerfile.streamlit
│   ├── docker-compose.yml
│   └── docker-compose.full-stack.yml
│
├── docs/                        # Documentation (local only)
│   ├── COMPLETE_DEMO_GUIDE.md   # Full system demo
│   ├── DEPLOYMENT_GUIDE.md      # Deployment instructions
│   └── IMPLEMENTATION_SUMMARY.md
│
├── data_clean/                  # Medallion data architecture
│   ├── bronze/                  # Raw data
│   ├── silver/                  # Engineered features
│   ├── gold/                    # Training-ready data
│   └── models/                  # Trained XGBoost models
│
├── feature_repo/                # Feast feature store config
├── airflow_mlops/               # Airflow DAGs & config
├── news-simulator/              # News article generator
└── archive/                     # Archived code & data
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
cd docker
docker compose -f docker-compose.full-stack.yml restart

# View logs
docker compose -f docker-compose.full-stack.yml logs -f <service>
```

## 📝 License

Educational and research purposes only.

---

**Version**: 2.0.0
**Python**: 3.11+
**Last Updated**: October 2025
