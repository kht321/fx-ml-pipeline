# S&P 500 ML Prediction Pipeline

> Production-ready, end-to-end machine learning pipeline for S&P 500 price prediction using technical analysis, market microstructure, and financial sentiment from news

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![MLflow](https://img.shields.io/badge/MLflow-3.5.0-green.svg)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.9.3-orange.svg)](https://airflow.apache.org/)
[![License](https://img.shields.io/badge/License-Educational-red.svg)](LICENSE)

## Overview

This project implements a complete MLOps pipeline for financial market prediction, demonstrating modern machine learning engineering best practices from data ingestion to model deployment and monitoring.

### Key Highlights

- **Production Architecture**: Medallion data pipeline (Bronze → Silver → Gold) with complete MLOps stack
- **Multi-Model Selection**: Automatic comparison of XGBoost, LightGBM, and ARIMAX with RMSE-based selection
- **Advanced Feature Engineering**: 114 features including technical indicators, market microstructure, volatility estimators, and AI-powered news sentiment
- **Optimized FinBERT Processing**: Batch inference (20-30x speedup) for financial sentiment analysis
- **Full Automation**: Airflow orchestration with daily retraining, automatic model selection, and deployment
- **Comprehensive Monitoring**: Evidently AI drift detection + email alerting for data drift and model degradation
- **Enhanced MLflow Integration**: Full model lifecycle management with staging, versioning, and promotion workflows
- **Real-time Inference**: FastAPI REST API + WebSocket streaming with <100ms latency
- **Interactive Dashboard**: Streamlit UI with live predictions, news snippets, forecast visualization, sentiment timeline, and model metrics
- **Containerized Deployment**: 16 Docker services with Blue/Green deployment and Nginx load balancing

---

## What's New in v4.0

### Multi-Model Training & Selection Pipeline
- **3 Models Compete**: XGBoost, LightGBM, and ARIMAX all trained in parallel
- **Fair Comparison**: All models use identical features (market + news signals)
- **Automatic Selection**: Best model selected based on test RMSE
- **ARIMAX Enhancement**: Now includes news features as exogenous variables (previously excluded)

### FinBERT Performance Optimization
- **Batch Processing**: Process 64 articles simultaneously (previously 1 at a time)
- **20-30x Speedup**: 10-15 minutes instead of 4.5 hours for 25K articles
- **GPU/CPU Efficient**: Optimized PyTorch batch inference
- **Fallback Handling**: Graceful degradation to single processing on errors

### Comprehensive Drift Detection (Evidently AI)
- **Data Drift Monitoring**: Kolmogorov-Smirnov test with configurable thresholds (default: 10%)
- **Performance Degradation**: RMSE increase alerts (default: 20% threshold)
- **Missing Values**: Alert when missing data exceeds 5%
- **Automated HTML Reports**: Generated automatically with visualizations
- **Email Integration**: Notifications sent when drift detected

### Enhanced MLflow Model Management
- **Model Versioning**: Automatic version tracking for all registered models
- **Stage Promotion Workflow**: None → Staging → Production → Archived lifecycle
- **Model Aliases**: "champion" and "challenger" labels for easy reference
- **Cross-Experiment Comparison**: Compare models across different experiments
- **Transition Logging**: Complete audit trail of model promotions

### Email Alerting System
- **SMTP Integration**: Gmail/custom SMTP support with app-specific passwords
- **HTML Formatted Emails**: Professional, easy-to-read notifications
- **File Attachments**: Drift reports automatically attached to alerts
- **Multiple Alert Types**: Drift detection, pipeline failures, status updates
- **Configurable Recipients**: Environment variable-based configuration

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 8GB RAM minimum (16GB+ recommended for full pipeline)
- macOS, Linux, or WSL2 on Windows

### 5-Minute Demo (Regression Model + News Sentiment)

```bash
# 1. Clone and setup
git clone https://github.com/kht321/fx-ml-pipeline.git
cd fx-ml-pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure email alerts (optional)
cp .env.monitoring.example .env.monitoring
# Edit .env.monitoring with your SMTP credentials

# 3. Start all services
docker-compose up -d

# 4. Access the services
# - Airflow: http://localhost:8080 (admin/admin)
# - MLflow: http://localhost:5001
# - Streamlit Dashboard: http://localhost:8501
# - FastAPI Docs: http://localhost:8000/docs
```

**What you get:**
- Multi-model training (XGBoost, LightGBM, ARIMAX)
- Automatic best model selection
- Real-time news sentiment analysis
- Drift detection and monitoring
- Email alerts on model degradation
- Interactive prediction dashboard
- REST API for programmatic access

### Stop All Services

```bash
docker-compose down
```

**For complete step-by-step instructions, see [docs/QUICKSTART.md](docs/QUICKSTART.md)**

---

## Performance Metrics

### Model Selection Results

**Latest Pipeline Run (3-Model Comparison):**

```
XGBoost Regression:
├─ Test RMSE:        0.0234
├─ Test MAE:         0.0187
├─ OOT RMSE:         0.0241
└─ Training Time:    3-4 minutes

LightGBM Regression:
├─ Test RMSE:        0.0229
├─ Test MAE:         0.0183
├─ OOT RMSE:         0.0238
└─ Training Time:    2-3 minutes

ARIMAX with News:
├─ Test RMSE:        0.0256
├─ Test MAE:         0.0198
├─ OOT RMSE:         0.0267
└─ Training Time:    5-6 minutes

Selected Model: LightGBM (lowest test RMSE)
```

### System Performance

```
Pipeline Stages:
├─ Data Validation:      30 seconds
├─ Silver Processing:    5-8 minutes (parallel)
├─ FinBERT Analysis:     10-15 minutes (optimized batch processing)
├─ Gold Processing:      3-5 minutes
├─ Model Training:       8-10 minutes (3 models in parallel)
├─ Model Selection:      10 seconds
├─ Deployment:           30 seconds
└─ Total Pipeline:       25-35 minutes

Data Volume:
├─ Market Candles:   1.7M+ (1-minute, 2-5 years)
├─ News Articles:    25K-100K (GDELT + RSS feeds)
├─ Features:         114 total
└─ Training Samples: ~1.4M (85% of data)

Optimization Results:
├─ FinBERT (Before):  4.5 hours (1.5 articles/sec)
├─ FinBERT (After):   10-15 minutes (30-45 articles/sec)
└─ Speedup:           20-30x faster
```

---

## Key Features

### 1. Multi-Model Selection Pipeline

**Automatic Model Competition:**
- **XGBoost**: Gradient boosting with tree-based learning
- **LightGBM**: Fast gradient boosting with leaf-wise growth
- **ARIMAX**: Time series with exogenous variables (news sentiment)

**Selection Criteria:**
- Primary metric: Test RMSE
- Fallback metrics: MAE, OOT performance
- Automatic deployment of best model to production
- Complete selection metadata saved (selection_info.json)

**Fair Comparison:**
- All models trained on identical features
- Same train/test/OOT splits
- Consistent preprocessing and scaling
- News signals integrated into all models (including ARIMAX)

### 2. Advanced Feature Engineering (114 Total Features)

#### Market Features (108 features)

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

**Optimization:** Batch processing (64 articles at once) for 20-30x speedup

### 3. Comprehensive Drift Detection (Evidently AI)

**Automated Monitoring:**

```python
# Data Drift Detection
- Kolmogorov-Smirnov test for feature distributions
- Configurable threshold (default: 10%)
- Per-feature drift scores and visualizations

# Performance Degradation
- RMSE increase monitoring (default: 20% threshold)
- Comparison: reference vs current data
- Automatic alert on significant degradation

# Missing Values Monitoring
- Alert when missing data exceeds 5%
- Critical feature tracking
- Data quality reports

# HTML Reports
- Automated generation after each run
- Interactive visualizations
- Feature importance and drift analysis
```

**Email Alerts:**
- Sent automatically when drift detected
- Includes detailed Evidently report as attachment
- Configurable recipients via `.env.monitoring`
- Professional HTML formatting with metrics table

**Configuration:** [.env.monitoring](.env.monitoring)

### 4. Enhanced MLflow Model Management

**Model Lifecycle Management:**

```python
# Model Registration
- Automatic versioning for all models
- Tags: model_type, performance_metrics, dataset_version
- Description and metadata tracking

# Stage Promotion Workflow
None → Staging → Production → Archived

# Promotion Triggers
- Automatic: Test RMSE below threshold
- Manual: Via MLflow UI
- Programmatic: Python API

# Model Aliases
- "champion": Current production model
- "challenger": Candidate for promotion
- Custom aliases for A/B testing

# Cross-Experiment Comparison
- Compare models across experiments
- Metric differences and percentage changes
- Version history tracking
```

**CLI Tools:**

```bash
# List all model versions
python -m src_clean.monitoring.mlflow_model_manager \
  --action list \
  --model-name sp500_best_model

# Promote model to staging
python -m src_clean.monitoring.mlflow_model_manager \
  --action promote-staging \
  --version 3

# Promote to production
python -m src_clean.monitoring.mlflow_model_manager \
  --action promote-prod \
  --version 3

# Compare two versions
python -m src_clean.monitoring.mlflow_model_manager \
  --action compare \
  --version 2 --version2 3

# View summary by stage
python -m src_clean.monitoring.mlflow_model_manager \
  --action summary
```

### 5. Email Alerting System

**Setup:**

1. For Gmail, generate app-specific password:
   - Visit: https://myaccount.google.com/apppasswords
   - Select "Mail" and generate password
   - Copy to `.env.monitoring`

2. Configure environment variables:

```bash
# .env.monitoring
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
ALERT_RECIPIENTS=recipient1@example.com,recipient2@example.com
```

**Alert Types:**

```python
# Drift Detection Alert
- Sent when data drift exceeds threshold
- Includes drift summary table
- Evidently HTML report attached
- Recommended actions listed

# Pipeline Status Alert
- Success/Failure/Warning notifications
- Execution details and metrics
- Error messages if applicable

# Test Email
python -m src_clean.monitoring.email_alerter \
  --to your_email@example.com \
  --greeting "Boss"
```

### 6. Real-Time Inference System

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
  "timestamp": "2025-11-01T19:45:00",
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

---

## Technology Stack

### ML & Data Science
| Technology | Version | Purpose |
|-----------|---------|---------|
| **XGBoost** | 3.0.5 | Gradient boosting classifier/regressor |
| **LightGBM** | Latest | Fast gradient boosting alternative |
| **ARIMAX** | statsmodels | Time series with exogenous variables |
| **FinBERT** | ProsusAI/finbert | Financial sentiment analysis (transformers) |
| **Scikit-learn** | 1.7.2 | Data preprocessing, CV, metrics |
| **Pandas** | 2.3.3 | Data manipulation |
| **NumPy** | 1.26.4 | Numerical computing |

### MLOps & Orchestration
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Apache Airflow** | 2.9.3 | Workflow orchestration, DAG scheduling |
| **MLflow** | 3.5.0 | Experiment tracking, model registry, versioning |
| **Feast** | 0.47.0 | Feature store (online/batch serving) |
| **Evidently AI** | Latest | Model monitoring, drift detection, HTML reports |

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

## Data Pipeline

### Bronze Layer (Raw Data)

**Market Data from OANDA:**
```
Source:  OANDA REST API (SPX500_USD)
Format:  NDJSON (newline-delimited JSON)
Schema:  {time, open, high, low, close, volume, bid, ask}
Volume:  1.7M+ 1-minute candles (2-5 years)
Storage: data_clean/bronze/market/spx500_usd_m1_*.ndjson
```

**News Data (Multiple Sources):**
```
Sources: GDELT Project (primary), RSS feeds, Alpha Vantage, Finnhub
Format:  JSON files
Schema:  {title, body, source, date, url, author, language}
Volume:  25,000-100,000 articles (5-year historical)
Storage: data_clean/bronze/news/historical_5year/*.json
```

### Silver Layer (Processed Features)

**Processing Pipeline:**
1. **Technical Features** → `market_technical_processor.py` (2-3 min)
2. **Microstructure** → `market_microstructure_processor.py` (1-2 min)
3. **Volatility** → `market_volatility_processor.py` (2-3 min)
4. **News Sentiment** → `news_sentiment_processor.py` (30 sec)

### Gold Layer (Training-Ready)

**Gold Processing:**
1. **Market Merge** → `market_gold_builder.py` (30 sec)
2. **FinBERT Signals** → `news_signal_builder.py` (10-15 min with batch optimization)
3. **Label Generation** → `label_generator.py` (1 min)

**Optimization Details:**
- Batch size: 64 articles per inference
- PyTorch batch processing with padding
- GPU support (automatically detected)
- Fallback to single processing on errors
- Progress bars with tqdm

**Total Pipeline Time:** 25-35 minutes for complete run

---

## Deployment & Monitoring

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

### Service Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5001 | No auth |
| FastAPI | http://localhost:8000/docs | No auth |
| Streamlit | http://localhost:8501 | No auth |
| Evidently | http://localhost:8050 | No auth |
| Nginx (Load Balancer) | http://localhost:8088 | No auth |

### Quick Commands

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f mlflow

# Restart specific service
docker-compose restart airflow-scheduler

# Stop all services
docker-compose down

# Clean slate (remove volumes)
docker-compose down -v
```

---

## Monitoring & Alerting

### Evidently AI Drift Detection

**Automated Checks:**
- Data drift (KS test, threshold: 10%)
- Performance degradation (RMSE increase > 20%)
- Missing values (threshold: 5%)
- Feature distribution changes

**Generated Reports:**
- HTML reports with interactive visualizations
- Per-feature drift scores
- Model performance comparison
- Data quality metrics

**Access:** http://localhost:8050

### Email Alerts

**Trigger Conditions:**
- Data drift detected above threshold
- Model performance degradation
- Pipeline execution failures
- Critical data quality issues

**Email Content:**
- Summary table with drift metrics
- Recommended actions
- Evidently HTML report attached
- System information and timestamps

**Configuration:** [.env.monitoring](.env.monitoring)

### MLflow Model Registry

**Model Lifecycle:**
```
None → Staging → Production → Archived

Transitions triggered by:
├─ Automatic: Test RMSE meets threshold
├─ Manual: Via MLflow UI
└─ Programmatic: Python API
```

**Tracking:**
- All model versions and experiments
- Performance metrics and parameters
- Artifacts (models, plots, reports)
- Stage transition history

### Health Checks

**Automated Validation:**
```python
# Model Performance
test_rmse < threshold
oot_rmse < 1.2 * test_rmse

# Data Quality
missing_values < 5%
outliers < 5 std deviations
drift_score < 0.1

# System Performance
inference_latency < 100ms
api_uptime > 99%
```

---

## Project Structure

```
fx-ml-pipeline/
├── README.md                    # This file
├── requirements.txt             # Python 3.11 dependencies
├── docker-compose.yml           # Unified orchestration (16 services)
├── .env.monitoring              # Email and monitoring configuration
│
├── src_clean/                   # Production code
│   ├── data_pipelines/          # Bronze → Silver → Gold
│   │   ├── bronze/              # Data collection
│   │   ├── silver/              # Feature engineering
│   │   └── gold/                # Training preparation
│   │       └── news_signal_builder.py  # FinBERT batch processing
│   ├── training/                # Model training
│   │   ├── xgboost_training_pipeline_mlflow.py
│   │   ├── lightgbm_training_pipeline_mlflow.py
│   │   └── arima_training_pipeline_mlflow.py  # ARIMAX with news
│   ├── monitoring/              # Drift detection & alerting
│   │   ├── evidently_drift_detector.py  # Comprehensive drift detection
│   │   ├── email_alerter.py             # Email notification system
│   │   └── mlflow_model_manager.py      # Model lifecycle management
│   ├── api/                     # FastAPI backend
│   ├── ui/                      # Streamlit dashboard
│   └── utils/                   # Shared utilities
│
├── airflow_mlops/               # Airflow orchestration
│   └── dags/
│       └── sp500_ml_pipeline_v4_docker.py  # Production DAG (17 tasks)
│
├── docs/                        # Documentation
│   ├── QUICKSTART.md            # Quick start guide
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── OPTIMIZATION_COMPLETE.md
│   ├── GMAIL_APP_PASSWORD_SETUP.md
│   └── ... (all other docs)
│
├── config/                      # Configuration files
│   └── drift_thresholds.json    # Drift detection settings
│
├── data_clean/                  # Medallion data
│   ├── bronze/                  # Raw data
│   ├── silver/                  # Processed features
│   └── gold/                    # Training-ready data
│
├── models/                      # Trained models
│   ├── xgboost/                 # XGBoost models
│   ├── lightgbm/                # LightGBM models
│   ├── arima/                   # ARIMAX models
│   └── production/              # Selected best model
│       ├── best_model_*.pkl
│       └── selection_info.json  # Model selection metadata
│
├── feature_repo/                # Feast feature definitions
├── mlruns/                      # MLflow experiment store
└── logs/                        # Application logs
```

---

## Demo & Documentation

**Quick Start Guides:**
- [Quick Start Guide](docs/QUICKSTART.md) - 5-minute setup
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md) - Complete feature overview
- [Optimization Guide](docs/OPTIMIZATION_COMPLETE.md) - FinBERT performance improvements

**Setup Guides:**
- [Gmail Setup](docs/GMAIL_APP_PASSWORD_SETUP.md) - Email alerting configuration
- [Airflow Guide](docs/AIRFLOW_SETUP_GUIDE.md) - Airflow configuration and usage

**Advanced Topics:**
- Model selection and comparison
- Drift detection configuration
- MLflow model lifecycle management
- Custom alert thresholds

---

## Common Issues & Fixes

### Port 5000 Conflict (macOS)
**Problem:** macOS ControlCenter uses port 5000 for AirPlay Receiver

**Solution:** MLflow configured to use port 5001
```bash
open http://localhost:5001
```

### Airflow Tasks Out of Memory
**Problem:** Docker containers killed (exit 137) due to OOM

**Solution:** Increase Docker Desktop memory allocation to 8GB+
```bash
# Docker Desktop → Settings → Resources → Memory: 8GB+
```

### FinBERT Processing Too Slow
**Problem:** News sentiment analysis taking hours

**Solution:** Already optimized with batch processing (v4.0)
- Batch size: 64 articles at once
- 20-30x faster than sequential processing
- Automatic GPU detection and usage

### Email Alerts Not Sending
**Problem:** SMTP authentication failures

**Solution:**
1. Use app-specific password for Gmail (not regular password)
2. Generate at: https://myaccount.google.com/apppasswords
3. Update `.env.monitoring` with app password
4. Test with: `python -m src_clean.monitoring.email_alerter --to your_email@example.com`

### Drift Detection Not Running
**Problem:** No drift reports generated

**Solution:**
```bash
# Manual drift detection test
python -m src_clean.monitoring.evidently_drift_detector \
  --reference-data data_clean/gold/market/features/spx500_features.csv \
  --current-data data_clean/gold/monitoring/current_features.csv \
  --alert-email your_email@example.com
```

---

## Support

**Documentation:**
- [Quick Start Guide](docs/QUICKSTART.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)
- [Optimization Guide](docs/OPTIMIZATION_COMPLETE.md)

**Troubleshooting:**
- Check logs: `logs/` directory
- Docker logs: `docker-compose logs -f <service>`
- Service health: `docker-compose ps`
- Airflow UI: http://localhost:8080
- MLflow UI: http://localhost:5001

**GitHub Issues:**
https://github.com/kht321/fx-ml-pipeline/issues

---

## License

Educational and research purposes only. Not financial advice.

---

## Acknowledgments

- **OANDA** for free market data API
- **GDELT Project** for unlimited historical news
- **ProsusAI** for FinBERT model
- **MLflow**, **Airflow**, **Feast**, **Evidently AI** communities

---

**Version:** 4.0 (Multi-Model Selection + Optimization)
**Python:** 3.11+
**Last Updated:** November 1, 2025
**Status:** Fully Operational
