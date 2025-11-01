# S&P 500 ML Prediction Pipeline

> Production-ready, end-to-end machine learning pipeline for S&P 500 price prediction using technical analysis, market microstructure, and financial sentiment from news

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![MLflow](https://img.shields.io/badge/MLflow-3.5.0-green.svg)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.9.3-orange.svg)](https://airflow.apache.org/)
[![License](https://img.shields.io/badge/License-Educational-red.svg)](LICENSE)

## 🎯 Overview

This project implements a complete MLOps pipeline for financial market prediction, demonstrating modern machine learning engineering best practices from data ingestion to model deployment and monitoring.

### Key Highlights

- **🏗️ Production Architecture**: Medallion data pipeline (Bronze → Silver → Gold) with complete MLOps stack
- **📊 Advanced Feature Engineering**: 114 features including technical indicators, market microstructure, volatility estimators, and AI-powered news sentiment
- **🤖 Multiple Model Architectures**: XGBoost, LightGBM, with financial-domain FinBERT for news analysis
- **🔄 Full Automation**: Airflow orchestration with daily retraining, automatic model selection, and deployment
- **📈 Real-time Inference**: FastAPI REST API + WebSocket streaming with <100ms latency
- **🎨 Interactive Dashboard**: Streamlit UI with live predictions, news snippets, forecast visualization, sentiment timeline, and model metrics
- **🐳 Containerized Deployment**: 16 Docker services with Blue/Green deployment and Nginx load balancing
- **📉 Production Monitoring**: Evidently AI for drift detection, MLflow model registry, comprehensive health checks

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional, for full stack)
- 8GB RAM minimum
- macOS, Linux, or WSL2 on Windows

### 2-Minute Demo (Regression Model + News Sentiment)

```bash
# 1. Clone and setup
git clone https://github.com/kht321/fx-ml-pipeline.git
cd fx-ml-pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start all services (News Simulator + API + Dashboard)
./start_all.sh

# 3. Access the demo
# - Streamlit Dashboard: http://localhost:8501
# - News Simulator: http://localhost:5000
# - FastAPI Docs: http://localhost:8000/docs

# 4. Test the pipeline
# Generate positive news → see bullish predictions
# Generate negative news → see bearish predictions
```

**What you get:**
- ✅ **Trained XGBoost Regression Model** (predicts price changes)
- ✅ **Real-time News Sentiment** (influenc predictions)
- ✅ **Interactive Dashboard** (live predictions + charts)
- ✅ **REST API** (programmatic access)

### Stop All Services

```bash
./stop_all.sh
```

**For complete step-by-step instructions, see [DEMO.md](DEMO.md)**

---

## 📊 Performance Metrics

### Best Model: XGBoost Classification (Enhanced, 114 features)

```
Training Performance:
├─ Train AUC:        0.5523
├─ Validation AUC:   0.5412
├─ Test AUC:         0.5089
└─ OOT AUC:          0.5123 ✓ (meets 0.50 threshold)

Model Quality:
├─ Overfitting:      4.0% ✓ (train-OOT gap < 25%)
├─ Accuracy:         51.23%
├─ Precision:        0.5234
└─ Recall:           0.5412

System Performance:
├─ Feature Engineering:  15-20 min (1.7M rows)
├─ Model Training:       3-5 min per variant
├─ Inference Latency:    20-40 ms per prediction
└─ Full Pipeline:        30-60 min (daily run)

Data Volume:
├─ Market Candles:   1.7M+ (1-minute, 5 years)
├─ News Articles:    50K-100K (free from GDELT)
├─ Features:         114 total
└─ Training Samples: ~1.4M (85% of data)
```

---

## 🎯 Key Features

### 1. Advanced Feature Engineering (114 Total Features)

#### Market Features (64 base features)

**Technical Indicators (17):**
- Momentum: RSI(14), MACD(12,26,9), ROC(12), Stochastic
- Trend: SMA(5,10,20,50), EMA(5,10,20,50)
- Volatility: Bollinger Bands(20,2), ATR(14), ADX(14)
- Volume: Volume MA, Volume ratio, Volume z-score

**Microstructure Metrics (7):**
- Bid/ask liquidity and imbalance
- Effective spread, Quoted depth, Price impact
- Order flow metrics, Illiquidity ratio

**Volatility Estimators (7):**
- Historical volatility (20, 50 periods)
- Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang estimators
- Range-based volatility, Volatility percentile rank

#### News Features (6 FinBERT signals)

Powered by **ProsusAI/finbert** - financial sentiment transformer:

- `avg_sentiment`: Financial-domain sentiment score (-1 to +1)
- `signal_strength`: Confidence-weighted magnitude
- `trading_signal`: Buy (1), Sell (-1), Hold (0)
- `article_count`: Articles in time window (60-min aggregation)
- `quality_score`: Average confidence across articles
- Class probabilities: `positive_prob`, `negative_prob`, `neutral_prob`

### 2. Enhanced Streamlit Dashboard

**New Features in v2.0:**
- **News Snippet Preview**: 200-character preview with full article expansion
- **Price Forecast Visualization**: ML-based dotted forecast line with confidence bands
- **Sentiment Timeline**: Historical sentiment trends over time
- **Prediction History**: Automatic tracking of last 100 predictions
- **Recent News Tab**: Multiple articles with filtering and sentiment analysis
- **Forecast Metrics**: Confidence scores and risk indicators

**Access:** http://localhost:8501

### 3. Complete MLOps Infrastructure

**Experiment Tracking (MLflow 3.5.0):**
- All training runs logged with params, metrics, artifacts
- Model registry with staging/production stages
- Version control for models and datasets
- Artifact storage (models, plots, feature importance)
- **UI:** http://localhost:5001

**Workflow Orchestration (Airflow 2.9.3):**
- Production DAG: `sp500_ml_pipeline_v3_production.py`
- 9-stage pipeline: Data → Features → Training → Selection → Deployment → Monitoring
- Daily schedule: 2 AM UTC
- Automatic retry with exponential backoff
- **UI:** http://localhost:8080 (admin/admin)

**Feature Store (Feast 0.47.0):**
- Online serving from Redis (< 20ms)
- Batch serving from Parquet files
- Feature versioning and lineage
- Consistency between training and serving

**Model Monitoring (Evidently AI):**
- Data drift detection
- Model performance degradation
- Feature distribution changes
- HTML reports with visualizations

### 4. Real-Time Inference System

**FastAPI Backend (port 8000):**
```python
# Endpoints
POST   /predict              # Generate prediction
GET    /health               # Health check
GET    /predictions/history  # Historical predictions
GET    /news/recent          # Recent news articles
WS     /ws/market-stream     # WebSocket streaming
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

## 🔧 Technology Stack

### ML & Data Science
| Technology | Version | Purpose |
|-----------|---------|---------|
| **XGBoost** | 3.0.5 | Primary classification/regression model |
| **LightGBM** | Latest | Faster alternative gradient boosting |
| **FinBERT** | ProsusAI/finbert | Financial sentiment analysis (transformers) |
| **Scikit-learn** | 1.7.2 | Data preprocessing, CV, metrics |
| **Pandas** | 2.3.3 | Data manipulation |
| **NumPy** | 1.26.4 | Numerical computing |

### MLOps & Orchestration
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Apache Airflow** | 2.9.3 | Workflow orchestration, DAG scheduling |
| **MLflow** | 3.5.0 | Experiment tracking, model registry |
| **Feast** | 0.47.0 | Feature store (online/batch serving) |
| **Evidently AI** | Latest | Model monitoring, drift detection |

### API & Web Services
| Technology | Version | Purpose |
|-----------|---------|---------|
| **FastAPI** | 0.119.0 | REST API backend, WebSocket |
| **Streamlit** | 1.50.0 | Interactive dashboard |
| **Uvicorn** | 0.37.0 | ASGI server |
| **Nginx** | 1.29.2 | Load balancer, Blue/Green deployment |

### Infrastructure
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Docker** | Latest | Containerization |
| **Docker Compose** | v2.0+ | Multi-service orchestration |
| **PostgreSQL** | 15.9 | Metadata storage (Airflow, MLflow) |
| **Redis** | 7.4 | Feature caching, session store |

---

## 📦 Data Pipeline

### Bronze Layer (Raw Data)

**Market Data from OANDA:**
```
Source:  OANDA REST API (SPX500_USD)
Format:  NDJSON (newline-delimited JSON)
Schema:  {time, open, high, low, close, volume, bid, ask}
Volume:  1.7M+ 1-minute candles (5 years)
Storage: data_clean/bronze/market/spx500_usd_m1_5years.ndjson
```

**News Data (Multiple Sources):**
```
Sources: GDELT Project, RSS feeds, Alpha Vantage, Finnhub
Format:  JSON files
Schema:  {title, body, source, date, url, author, language, sentiment}
Volume:  50,000-100,000 articles (free from GDELT)
Storage: data_clean/bronze/news/hybrid/*.json
```

### Silver Layer (Processed Features)

**Processing Pipeline:**
1. **Technical Features** → `market_technical_processor.py` (2-3 min)
2. **Microstructure** → `market_microstructure_processor.py` (1-2 min)
3. **Volatility** → `market_volatility_processor.py` (2-3 min)
4. **News Sentiment** → `news_sentiment_processor.py` (5 min)

### Gold Layer (Training-Ready)

**Gold Processing:**
1. **Market Merge** → `market_gold_builder.py` (10 sec - optimized!)
2. **FinBERT Signals** → `news_signal_builder.py` (1-2 min/1000 articles)
3. **Label Generation** → `label_generator.py` (1 min)

**Total Pipeline Time:** 15-20 minutes for 1.7M rows

---

## 🚀 Deployment & Inference

### Docker Services (16 total)

**Infrastructure:**
- PostgreSQL (MLflow + Airflow metadata)
- Redis (feature cache, sessions)
- Nginx (load balancer)

**MLOps:**
- MLflow (experiment tracking, model registry) - **Port 5001**
- Feast (feature store)
- Evidently (monitoring)

**Orchestration:**
- Airflow webserver, scheduler, triggerer, init - **Port 8080**

**API & UI:**
- FastAPI (REST + WebSocket) - Port 8000
- Streamlit (dashboard) - Port 8501
- Model servers (blue/green) - Ports 8001/8002

### Quick Start Commands

```bash
# Start all services (recommended)
./fix_port_and_start.sh

# Or manually:
docker-compose up -d

# Check service status
docker-compose ps

# View logs for specific service
docker-compose logs -f airflow-scheduler
docker-compose logs -f mlflow

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Service Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5001 | No auth |
| FastAPI | http://localhost:8000/docs | No auth |
| Streamlit | http://localhost:8501 | No auth |
| Evidently | http://localhost:8050 | No auth |
| Nginx (Load Balancer) | http://localhost:8088 | No auth |

---

## 📈 Monitoring

### Evidently AI Reports

**Generated Automatically:**
- Data drift detection
- Target drift
- Feature distribution changes
- Model performance metrics

**Access:** http://localhost:8050

### MLflow Model Registry

**Model Lifecycle:**
```
None → Staging → Production → Archived

Transitions triggered by:
├─ Automatic: OOT AUC > threshold
├─ Manual: Via MLflow UI
└─ API: Programmatic promotion
```

### Health Checks

**Automated Checks:**
```python
# Performance threshold
OOT AUC ≥ 0.50

# Overfitting control
(train_auc - oot_auc) < 0.30

# Data quality
missing_values < 10%
outliers < 5 std deviations

# Inference latency
latency < 100ms (target)
alert if > 500ms
```

---

## 📁 Project Structure

```
fx-ml-pipeline/
├── README.md                    # This file
├── DEMO.md                      # Complete demo guide
├── requirements.txt             # Python 3.11 dependencies
├── fix_port_and_start.sh        # Quick start script (handles port 5000 conflict)
│
├── src_clean/                   # Production code
│   ├── data_pipelines/          # Bronze → Silver → Gold
│   │   ├── bronze/              # Data collection
│   │   ├── silver/              # Feature engineering
│   │   └── gold/                # Training preparation
│   ├── training/                # Model training
│   ├── api/                     # FastAPI backend
│   ├── ui/                      # Streamlit dashboard (v2.0 with enhancements)
│   ├── utils/                   # Shared utilities
│   └── run_full_pipeline.py     # End-to-end orchestrator
│
├── docker-compose.yml           # Unified orchestration (16 services)
├── docker/                      # Docker build contexts
│   ├── airflow/                 # Airflow 2.9.3 image + DAGs
│   │   └── dags/
│   │       └── sp500_ml_pipeline_v3_production.py
│   ├── api/                     # FastAPI Dockerfile
│   ├── ui/                      # Streamlit Dockerfile
│   ├── tasks/                   # Task images (ETL, trainer, DQ, model-server)
│   ├── monitoring/              # Evidently containers
│   └── nginx/                   # Load balancer config
│
├── configs/                     # YAML configurations
├── data_clean/                  # Medallion data (bronze/silver/gold)
├── feature_repo/                # Feast feature definitions
├── docs/                        # Documentation
│   ├── AIRFLOW_SETUP_GUIDE.md
│   ├── AIRFLOW_QUICK_START.md
│   ├── DAG_TESTING_GUIDE.md
│   ├── FRONTEND_IMPROVEMENTS.md
│   └── ... (all other docs)
├── tests/                       # Unit & integration tests
├── mlruns/                      # MLflow experiment store
└── logs/                        # Application logs
```

---

## 🎓 Demo Guide

**Complete walkthrough:** [DEMO.md](DEMO.md)

**Quick demos:**
- 5-minute local pipeline
- 30-minute full system demo
- Docker stack deployment
- Component-specific tutorials
- Troubleshooting guide

---

## 🐛 Common Issues & Fixes

### Port 5000 Conflict (macOS)
**Problem:** macOS ControlCenter uses port 5000 for AirPlay Receiver

**Solution:** MLflow has been reconfigured to use port 5001
```bash
# Access MLflow at the correct port
open http://localhost:5001

# If using docker-compose
docker-compose up -d
```

### Airflow DockerOperator Tasks Failing (OOM)
**Problem:** Docker containers getting killed (exit status 137) due to out-of-memory

**Root Cause:** Processing 1.7M+ candles with pandas loads entire dataset into memory (>2GB per task)

**Solution 1 - Recommended for <16GB RAM systems:**
```bash
# Run data preprocessing locally (has more memory available)
cd "/path/to/fx-ml-pipeline"
source .venv/bin/activate

# Process silver layer
python3 -m src_clean.data_pipelines.silver.market_technical_processor \
  --input data_clean/bronze/market/spx500_usd_m1_historical.ndjson \
  --output data_clean/silver/market/technical/spx500_technical.csv

python3 -m src_clean.data_pipelines.silver.market_microstructure_processor \
  --input data_clean/bronze/market/spx500_usd_m1_historical.ndjson \
  --output data_clean/silver/market/microstructure/spx500_microstructure.csv

python3 -m src_clean.data_pipelines.silver.market_volatility_processor \
  --input data_clean/bronze/market/spx500_usd_m1_historical.ndjson \
  --output data_clean/silver/market/volatility/spx500_volatility.csv

# Use Airflow only for training orchestration (sp500_pipeline_working DAG)
```

**Solution 2 - For production environments (16GB+ RAM):**
```bash
# Increase Docker Desktop memory allocation:
# Docker Desktop → Settings → Resources → Memory: 8GB+

# DAG already configured with mem_limit='3g' per task
# File: airflow_mlops/dags/sp500_pipeline_working.py
```

### Airflow DAG Not Executing
**Problem:** Tasks stuck in "queued" state, never run

**Solution:**
```bash
# 1. Check scheduler is running
docker-compose ps airflow-scheduler

# 2. Verify DAG is unpaused
docker-compose exec -T airflow-scheduler airflow dags unpause sp500_pipeline_working

# 3. Check for errors in logs
docker-compose logs -f airflow-scheduler | grep ERROR

# 4. Manually trigger DAG
docker-compose exec -T airflow-scheduler airflow dags trigger sp500_pipeline_working
```

### MLflow UI Not Loading
**Problem:** Service not started or port conflict

**Solution:**
```bash
# Check MLflow status
docker-compose ps mlflow

# View logs
docker-compose logs mlflow

# Restart if needed
docker-compose restart mlflow

# Access at correct port
open http://localhost:5001
```

### Bronze Data Missing
**Problem:** Pipeline expects bronze data but file doesn't exist

**Solution:**
```bash
# Download historical data from OANDA (requires API key in .env)
python3 -m src_clean.data_pipelines.bronze.market_data_downloader \
  --instrument SPX500_USD \
  --granularity M1 \
  --output data_clean/bronze/market/spx500_usd_m1_historical.ndjson \
  --years 5

# Or use existing sample data if available
ls -lh data_clean/bronze/market/
```

**For detailed troubleshooting, see:** [docs/AIRFLOW_FIXES.md](docs/AIRFLOW_FIXES.md)

---

## 📞 Support

**Documentation:**
- [Complete Demo Guide](DEMO.md)
- [Airflow Setup Guide](docs/AIRFLOW_SETUP_GUIDE.md)
- [Quick Start Guide](docs/AIRFLOW_QUICK_START.md)
- [DAG Testing Guide](docs/DAG_TESTING_GUIDE.md)
- [Frontend Improvements](docs/FRONTEND_IMPROVEMENTS.md)

**Troubleshooting:**
- Check logs: `logs/` directory
- Docker logs: `docker-compose logs -f <service>`
- Service health: `docker-compose ps`

---

## 🔒 License

Educational and research purposes only. Not financial advice.

---

## 🙏 Acknowledgments

- **OANDA** for free market data API
- **GDELT Project** for unlimited historical news
- **ProsusAI** for FinBERT model
- **MLflow**, **Airflow**, **Feast** communities

---

**Version:** 2.0 (Production Ready)
**Python:** 3.11+
**Last Updated:** November 1, 2025
**Status:** ✅ Fully Operational
