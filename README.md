# S&P 500 ML Prediction Pipeline

> Machine Learning pipeline for S&P 500 market prediction using technical analysis and news sentiment

**Last Updated**: October 13, 2025

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Features](#features)
- [News Collection](#news-collection)
- [Model Training](#model-training)
- [Project Status](#project-status)
- [Documentation](#documentation)

---

## Overview

A production-ready ML pipeline for predicting S&P 500 movements using:
- **Market Data**: 5 years of 1-minute candles (1.7M+ samples) from OANDA
- **News Sentiment**: Financial news analysis powered by FinGPT
- **Dual Medallion Architecture**: Separate Bronze → Silver → Gold pipelines for market and news data
- **144 Features**: Technical indicators, microstructure signals, volatility estimators, and sentiment scores

### Key Metrics

| Metric | Value |
|--------|-------|
| **Market Data** | 1,705,276 candles (5 years) |
| **Data Size** | 353 MB |
| **Resolution** | 1-minute (M1) |
| **Market Features** | 37 features (Gold layer) |
| **News Features** | 11 features (Gold layer) |
| **Total Features** | 144 (across all layers) |
| **Processing Time** | ~2.5 minutes (full pipeline) |

---

## Quick Start

Choose between **Local Setup** (Python virtual environment) or **Docker** (containerized environment).

### Option A: Local Setup

#### 1. Setup Environment

```bash
# Clone repository
cd fx-ml-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

#### 2. Configure API Credentials

```bash
# Create .env file
cp .env.example .env

# Add your OANDA API credentials
echo "OANDA_TOKEN=your_token_here" >> .env
echo "OANDA_ACCOUNT_ID=your_account_id" >> .env
```

Get free OANDA credentials at: https://developer.oanda.com/

#### 3. Download S&P 500 Data

```bash
# Download 5 years of 1-minute S&P 500 data
python src/download_sp500_historical.py --years 5 --granularity M1

# Or use the convenience script
./scripts/download_sp500_data.sh --years 5
```

**Expected output**: `data/bronze/prices/spx500_usd_m1_5years.ndjson` (353 MB, ~1.7M candles)

#### 4. Run Market Pipeline

```bash
# Process market data: Bronze → Silver → Gold
python run_sp500_pipeline.py --skip-labels

# Output: 37 training-ready features in data/sp500/gold/
```

#### 5. Collect News Data (Optional)

```bash
# Option A: Free RSS feeds (no API keys required)
python src/scrape_sp500_news_free.py

# Option B: With API keys (better coverage)
./setup_news_api_keys.sh  # Interactive setup
python src/scrape_historical_sp500_news.py --recent-only

# Process news: Bronze → Silver → Gold
python src/build_news_features.py
python src/build_news_gold.py
```

#### 6. Train Models

```bash
# Train XGBoost models on combined features
python src/train_combined_model.py
```

### Option B: Docker Setup 🐳

**Recommended for**: Easy setup, reproducible environments, team collaboration

#### 1. Prerequisites

- Docker Desktop (Mac/Windows) or Docker Engine (Linux)
- Get from: https://www.docker.com/products/docker-desktop

#### 2. Quick Start with Docker

```bash
# Clone repository
git clone https://github.com/[your-username]/fx-ml-pipeline.git
cd fx-ml-pipeline

# Configure credentials
cp .env.example .env
nano .env  # Add your OANDA_TOKEN and OANDA_ACCOUNT_ID

# Build Docker images
docker-compose build

# Download data (runs in container)
docker-compose run --rm downloader

# Process data
docker-compose run --rm pipeline

# Collect and process news
docker-compose run --rm news-scraper
docker-compose run --rm news-processor

# Train models
docker-compose run --rm trainer
```

#### 3. Docker Services Available

| Service | Command | Purpose |
|---------|---------|---------|
| **dev** | `docker-compose run --rm dev` | Interactive development shell |
| **jupyter** | `docker-compose up -d jupyter` | JupyterLab (port 8888) |
| **downloader** | `docker-compose run --rm downloader` | Download S&P 500 data |
| **pipeline** | `docker-compose run --rm pipeline` | Run market pipeline |
| **news-scraper** | `docker-compose run --rm news-scraper` | Scrape news articles |
| **news-processor** | `docker-compose run --rm news-processor` | Process news data |
| **trainer** | `docker-compose run --rm trainer` | Train ML models |
| **api** | `docker-compose up -d api` | Model serving API (port 8000) |

#### 4. Docker Examples

```bash
# Interactive development
docker-compose run --rm dev /bin/bash
# Inside container, you have full access to the codebase

# Start Jupyter Lab
docker-compose up -d jupyter
# Access at: http://localhost:8888

# Run complete workflow
docker-compose run --rm downloader && \
docker-compose run --rm pipeline && \
docker-compose run --rm trainer

# View logs
docker-compose logs -f pipeline

# Stop all services
docker-compose down
```

**Full Docker documentation**: See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for comprehensive guide.

---

## Architecture

### Dual Medallion Design

```
┌─────────────────┐
│  Market Data    │
│  (OANDA API)    │
│  SPX500_USD     │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Bronze  │  Raw 1-min candles (NDJSON)
    └────┬────┘
         │
    ┌────▼────┐
    │ Silver  │  Technical + Microstructure + Volatility
    └────┬────┘
         │
    ┌────▼────┐
    │  Gold   │  37 training-ready features
    └────┬────┘
         │
         ├──────────┐
         │          │
    ┌────▼────┐ ┌──▼──────┐
    │ Models  │ │ Serving │
    └─────────┘ └─────────┘


┌─────────────────┐
│   News Data     │
│  (RSS Feeds +   │
│   News APIs)    │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Bronze  │  Raw articles (JSON)
    └────┬────┘
         │
    ┌────▼────┐
    │ Silver  │  Sentiment + Entities + Topics
    └────┬────┘
         │
    ┌────▼────┐
    │  Gold   │  11 trading signals
    └────┬────┘
         │
         └──────────┘
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Data Source** | OANDA v20 API (SPX500_USD CFD) |
| **Storage** | NDJSON (Bronze), CSV/Parquet (Silver/Gold) |
| **Feature Engineering** | Pandas, TA-Lib, NumPy |
| **NLP/Sentiment** | FinGPT (LLaMA 2 + LoRA adapters) |
| **ML Framework** | XGBoost, LightGBM |
| **Orchestration** | Python scripts, Airflow (planned) |
| **Feature Store** | Feast (local Parquet + Redis) |
| **Monitoring** | Prometheus + Grafana (planned) |

---

## Data Pipeline

### Market Pipeline: Bronze → Silver → Gold

#### Bronze Layer (Raw Data)
- **Format**: NDJSON (newline-delimited JSON)
- **Location**: `data/bronze/prices/`
- **Sample**: 1,705,276 candles
- **Fields**: `time`, `open`, `high`, `low`, `close`, `volume`

#### Silver Layer (Feature Engineering)

**Three parallel feature sets:**

1. **Technical Features** (`data/sp500/silver/technical_features/`)
   - RSI (14, 20 periods)
   - MACD (12, 26, 9)
   - Bollinger Bands
   - Moving averages (7, 14, 21, 50)
   - ATR, ADX
   - 17 features total

2. **Microstructure Features** (`data/sp500/silver/microstructure/`)
   - Bid-ask spread proxies
   - Price impact
   - Order flow imbalance
   - High-low range
   - 10 features total

3. **Volatility Features** (`data/sp500/silver/volatility/`)
   - Garman-Klass estimator
   - Parkinson estimator
   - Rogers-Satchell estimator
   - Yang-Zhang estimator
   - Historical volatility
   - 10 features total

#### Gold Layer (Training-Ready)
- **Location**: `data/sp500/gold/training/`
- **Format**: CSV with aligned timestamps
- **Features**: 37 columns (merged from all Silver sets)
- **Ready for**: Model training, backtesting, serving

### News Pipeline: Bronze → Silver → Gold

#### Bronze Layer
- **Location**: `data/news/bronze/raw_articles/`
- **Format**: Individual JSON files
- **Sources**: Yahoo Finance, CNBC, MarketWatch, Seeking Alpha
- **Current**: 27 recent articles

#### Silver Layer

1. **Sentiment Scores** (`data/news/silver/sentiment_scores/`)
   - Lexicon-based sentiment
   - FinGPT-enhanced sentiment (optional)
   - Confidence scores
   - Policy tone (hawkish/dovish)

2. **Entity Features** (`data/news/silver/entity_mentions/`)
   - Currency mentions
   - Key entity extraction
   - Text statistics

3. **Topic Signals** (`data/news/silver/topic_signals/`)
   - Topic categorization
   - Relevance scoring

#### Gold Layer
- **Location**: `data/news/gold/news_signals/`
- **Format**: CSV with hourly trading signals
- **Features**: 25 columns including:
  - Aggregated sentiment
  - Signal strength/direction
  - Quality scores
  - Time-decayed signals

---

## Features

### Complete Feature Inventory

See [outputs/COMPLETE_FEATURE_INVENTORY_WITH_FILES.csv](outputs/COMPLETE_FEATURE_INVENTORY_WITH_FILES.csv) for full feature catalog.

**Feature Breakdown by Category:**

| Category | Count | Description |
|----------|-------|-------------|
| **Price Action** | 4 | OHLC (Open, High, Low, Close) |
| **Technical** | 17 | RSI, MACD, Bollinger, MA, ATR, ADX |
| **Microstructure** | 10 | Spread, impact, imbalance, range |
| **Volatility** | 10 | GK, Parkinson, RS, YZ estimators |
| **Sentiment** | 11 | News sentiment, confidence, policy tone |
| **Meta** | 5 | Timestamps, volume, article counts |
| **Signals** | 14 | Trading signals, quality scores |

**Total**: 144 features across all pipeline layers

### Feature Quality

- **No missing values** in Gold layer (forward-fill applied)
- **Timestamp alignment** across all features
- **Normalized scales** ready for ML consumption
- **Time-series aware** splitting (no future leakage)

---

## News Collection

### Current Status
- **Collected**: 27 recent articles (Oct 1-12, 2025)
- **Method**: Free RSS feeds (no API keys required)
- **Processing**: Complete Bronze → Silver → Gold pipeline tested

### Options for Historical Data

| Method | Coverage | Cost | Setup |
|--------|----------|------|-------|
| **Daily RSS Collection** | Builds over time | Free | `cron` job |
| **Free APIs** | 30 days | Free | API keys required |
| **Paid APIs** | 5+ years | $50-450/mo | Subscription |
| **Alternative Sources** | Varies | Free | SEC EDGAR, Fed data |

### Quick News Collection

```bash
# Free RSS scraper (no API keys)
python src/scrape_sp500_news_free.py

# With API keys (better quality)
./setup_news_api_keys.sh
python src/scrape_historical_sp500_news.py --recent-only

# Process collected news
python src/build_news_features.py
python src/build_news_gold.py
```

See [docs/NEWS_COLLECTION_SUMMARY.md](docs/NEWS_COLLECTION_SUMMARY.md) for detailed guide.

---

## Model Training

### Training Pipeline

```bash
# 1. Prepare features (already done if pipeline ran)
python run_sp500_pipeline.py

# 2. Train models
python src/train_combined_model.py \
  --market-features data/sp500/gold/training/sp500_features.csv \
  --news-features data/news/gold/news_signals/sp500_trading_signals.csv \
  --output models/
```

### Model Configuration

**Default settings:**
- **Algorithm**: XGBoost with time-series splits
- **Validation**: Rolling window (train on past, test on future)
- **Hyperparameters**: Grid search on learning_rate, max_depth, n_estimators
- **Features**: Combined market (37) + news (11) = 48 features

### Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Regression**: MAE, RMSE, R²
- **Financial**: Sharpe ratio, max drawdown, win rate

---

## Project Status

### ✅ Completed Components

| Component | Status | Details |
|-----------|--------|---------|
| **Data Download** | ✅ | 5 years, 1.7M candles, 353 MB |
| **Market Pipeline** | ✅ | Bronze → Silver → Gold (37 features) |
| **News Pipeline** | ✅ | Bronze → Silver → Gold (11 features) |
| **Feature Engineering** | ✅ | 144 total features documented |
| **Pipeline Testing** | ✅ | Full E2E validation complete |
| **Documentation** | ✅ | Comprehensive guides created |

### 🚧 In Progress

| Component | Status | Next Steps |
|-----------|--------|------------|
| **News Historical Data** | 🚧 | Collect 5 years (see guide) |
| **Model Training** | 🚧 | XGBoost with combined features |
| **Model Evaluation** | 🚧 | Backtesting, metrics, feature importance |

### 📋 Planned

| Component | Priority | Description |
|-----------|----------|-------------|
| **Real-time Serving** | High | WebSocket API for live predictions |
| **Monitoring** | High | Prometheus + Grafana dashboards |
| **Airflow DAGs** | Medium | Scheduled pipeline orchestration |
| **Docker Deployment** | Medium | Containerization |
| **Multi-asset Support** | Low | Other indices (Nasdaq, Dow) |

---

## Documentation

### Quick Reference

| Document | Description |
|----------|-------------|
| **README.md** (this file) | Main project documentation |
| [SP500_PIPELINE_DOCUMENTATION.md](docs/SP500_PIPELINE_DOCUMENTATION.md) | Technical pipeline details |
| [FEATURE_INVENTORY_SUMMARY.md](docs/FEATURE_INVENTORY_SUMMARY.md) | Complete feature catalog |
| [NEWS_COLLECTION_SUMMARY.md](docs/NEWS_COLLECTION_SUMMARY.md) | News data collection guide |
| [NEWS_SCRAPING_GUIDE.md](docs/NEWS_SCRAPING_GUIDE.md) | News API setup instructions |
| [DATA_SHARING_GUIDE.md](docs/DATA_SHARING_GUIDE.md) | Large file sharing options |

### Test Results & Reports

| Report | Purpose |
|--------|---------|
| [NEWS_PIPELINE_TEST_RESULTS.md](docs/NEWS_PIPELINE_TEST_RESULTS.md) | News pipeline validation |
| [DATA_DOWNLOAD_RESULTS.md](docs/DATA_DOWNLOAD_RESULTS.md) | Market data download summary |
| [COMPLETE_PIPELINE_REPORT.md](docs/COMPLETE_PIPELINE_REPORT.md) | End-to-end pipeline report |

### Feature Inventory

**CSV Format**: [outputs/COMPLETE_FEATURE_INVENTORY_WITH_FILES.csv](outputs/COMPLETE_FEATURE_INVENTORY_WITH_FILES.csv)

Contains 51 entries documenting:
- Feature name and data type
- Category (technical, microstructure, volatility, sentiment)
- Source transformation and formula
- File location and format
- Sample values and status

---

## Directory Structure

```
fx-ml-pipeline/
├── data/                           # Data storage (git-ignored)
│   ├── bronze/prices/             # Raw S&P 500 candles (NDJSON)
│   ├── news/                      # News data
│   │   ├── bronze/                # Raw articles (JSON)
│   │   ├── silver/                # Sentiment features (CSV)
│   │   └── gold/                  # Trading signals (CSV)
│   └── sp500/                     # S&P 500 pipeline
│       ├── bronze/                # Raw candles (symlink)
│       ├── silver/                # Engineered features (CSV)
│       │   ├── technical_features/
│       │   ├── microstructure/
│       │   └── volatility/
│       └── gold/training/         # Training-ready (CSV)
│
├── src/                           # Source code
│   ├── download_sp500_historical.py
│   ├── build_market_features_from_candles.py
│   ├── build_sp500_gold.py
│   ├── scrape_sp500_news_free.py
│   ├── build_news_features.py
│   ├── build_news_gold.py
│   └── train_combined_model.py
│
├── scripts/                       # Convenience scripts
│   └── download_sp500_data.sh
│
├── docs/                          # Documentation (git-ignored)
│   ├── SP500_PIPELINE_DOCUMENTATION.md
│   ├── FEATURE_INVENTORY_SUMMARY.md
│   ├── NEWS_COLLECTION_SUMMARY.md
│   └── NEWS_SCRAPING_GUIDE.md
│
├── outputs/                       # Analysis outputs
│   └── COMPLETE_FEATURE_INVENTORY_WITH_FILES.csv
│
├── run_sp500_pipeline.py          # Main pipeline orchestrator
├── convert_ndjson_to_json_files.py
├── setup_news_api_keys.sh
├── .env.example                   # Environment template
├── .gitignore
├── requirements.txt
└── README.md                      # This file
```

---

## Getting Help

### Troubleshooting

**Issue**: "OANDA API rate limit exceeded"
- **Solution**: Use `--rate-limit-delay 0.3` or higher

**Issue**: "No news articles collected"
- **Solution**: Check API keys in `.env` or use free RSS scraper

**Issue**: "Out of memory during FinGPT processing"
- **Solution**: Use lexicon mode (default) or ensure 16GB+ RAM

**Issue**: "Git push rejected (large files)"
- **Solution**: Data files are git-ignored; see [docs/DATA_SHARING_GUIDE.md](docs/DATA_SHARING_GUIDE.md)

### Common Tasks

**Download more data**:
```bash
python src/download_sp500_historical.py --years 10
```

**Reprocess specific layer**:
```bash
# Rebuild Silver features only
python src/build_market_features_from_candles.py --input data/bronze/prices/spx500_usd_m1_5years.ndjson
```

**Update news collection**:
```bash
# Daily cron job
0 */6 * * * cd /path/to/fx-ml-pipeline && python src/scrape_sp500_news_free.py
```

### Support

For questions or issues:
1. Check [Documentation](#documentation) section
2. Review [docs/](docs/) folder for detailed guides
3. Check code comments in [src/](src/)
4. Contact project team

---

## Team & Contributions

| Area | Responsibilities |
|------|-----------------|
| **Data Engineering** | OANDA API integration, data collection |
| **Feature Engineering** | Technical indicators, sentiment analysis |
| **ML Pipeline** | Model training, evaluation, serving |
| **NLP/FinGPT** | News sentiment, entity extraction |
| **DevOps** | Orchestration, monitoring, deployment |

---

## License

This project is for educational and research purposes.

OANDA is a registered trademark. This project uses the OANDA v20 API under their terms of service.

---

## Changelog

### 2025-10-13
- ✅ Added Docker support (Dockerfile, docker-compose.yml, .dockerignore)
- ✅ Created comprehensive Docker guide (DOCKER_GUIDE.md)
- ✅ Multi-stage Docker builds (dev, production, jupyter, api)
- ✅ Docker services for all pipeline components
- ✅ Updated README with Docker quick start instructions
- ✅ Completed news pipeline test run (27 articles → 519 signals)
- ✅ Repository cleanup (removed old forex files)
- ✅ Consolidated documentation into comprehensive README
- ✅ Created feature inventory CSV (144 features documented)

### 2025-10-12
- ✅ Downloaded 5 years of S&P 500 data (1.7M candles, 353 MB)
- ✅ Built complete market pipeline (Bronze → Silver → Gold)
- ✅ Generated 37 training-ready features
- ✅ Created pipeline documentation and guides

### 2025-10-11
- ✅ Transformed from forex (SGD/USD) to S&P 500 pipeline
- ✅ Updated data download scripts for S&P 500
- ✅ Fixed timestamp parsing and timezone issues

---

**Project Status**: Production-Ready (Data & Features) | Model Training In Progress

**Repository**: https://github.com/[your-username]/fx-ml-pipeline (update with actual URL)

**Last Updated**: 2025-10-13
