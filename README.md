# S&P 500 ML Prediction Pipeline

Machine learning pipeline for S&P 500 price prediction using technical indicators and news sentiment.

## Quick Start

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Migrate to clean structure
python migrate_to_clean_structure.py --execute

# 3. Run full pipeline
python src_clean/run_full_pipeline.py \
  --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
  --skip-news \
  --prediction-horizon 30
```

## Architecture

**Medallion Data Pipeline**:
```
Bronze (Raw)  →  Silver (Features)  →  Gold (Training)  →  Models
```

- **Bronze**: Raw OHLCV candles + news articles
- **Silver**: Technical indicators (37 features)
- **Gold**: Training-ready features + labels
- **Models**: XGBoost classifier/regressor

## Components

### Data Pipeline (`src_clean/data_pipelines/`)

**Bronze Layer** - Data Collection:
- `bronze/market_data_downloader.py` - OANDA market data
- `bronze/news_data_collector.py` - RSS news feeds

**Silver Layer** - Feature Engineering:
- `silver/market_technical_processor.py` - RSI, MACD, Bollinger Bands
- `silver/market_microstructure_processor.py` - Volume, spread, order flow
- `silver/market_volatility_processor.py` - GK, Parkinson, RS, YZ estimators
- `silver/news_sentiment_processor.py` - Sentiment analysis

**Gold Layer** - Training Data:
- `gold/market_gold_builder.py` - Merge all market features
- `gold/label_generator.py` - Generate prediction labels

### Training (`src_clean/training/`)

- `xgboost_training_pipeline.py` - XGBoost model training with CV

### Orchestration

- `src_clean/run_full_pipeline.py` - End-to-end automation

## Implementation Status

| Component | Status |
|-----------|--------|
| Data structure (Bronze/Silver/Gold) | ✅ Complete |
| Market data collection | ✅ Complete |
| News data collection | ✅ Complete (RSS only) |
| Technical features (17) | ✅ Complete |
| Microstructure features (10) | ✅ Complete |
| Volatility features (10) | ✅ Complete |
| Sentiment features | ✅ Complete |
| Gold layer builder | ✅ Complete |
| Label generator (30min prediction) | ✅ Complete |
| XGBoost training pipeline | ✅ Complete |
| Full pipeline orchestrator | ✅ Complete |
| Historical news data (5 years) | ⚠️ In Progress |

## Data

### Market Data
- **Source**: OANDA SPX500_USD CFD
- **Coverage**: 5 years (2020-2025)
- **Records**: 1.7M candles
- **Resolution**: 1-minute

### News Data
- **Current**: 27 articles (12 days)
- **Required**: ~9,000 articles (5 years)
- **Status**: Daily collection ongoing

## Features

**Market Features (37)**:
- Technical: RSI, MACD, Bollinger Bands, Moving Averages, ATR, ADX
- Microstructure: Volume patterns, spread proxies, order flow
- Volatility: GK, Parkinson, Rogers-Satchell, Yang-Zhang

**News Features (11)**:
- Sentiment scores, polarity, confidence
- Policy tone (hawkish/dovish)
- Financial sentiment

## Prediction Target

**30-minute price prediction**:
- Classification: Price up (1) or down (0)
- Regression: Actual price change

## Key Scripts

```bash
# Analyze data quality
python analyze_data_coverage.py

# Process individual layers
python src_clean/data_pipelines/silver/market_technical_processor.py --input ... --output ...
python src_clean/data_pipelines/gold/market_gold_builder.py --technical ... --output ...
python src_clean/data_pipelines/gold/label_generator.py --input ... --output ... --horizon 30

# Train model
python src_clean/training/xgboost_training_pipeline.py \
  --market-features data_clean/gold/market/features/spx500_features.csv \
  --prediction-horizon 30 \
  --task classification
```

## Directory Structure

```
├── src_clean/           # Clean organized codebase
│   ├── data_pipelines/  # Bronze → Silver → Gold
│   └── training/        # Model training
├── data_clean/          # Medallion data structure
│   ├── bronze/          # Raw data
│   ├── silver/          # Engineered features
│   ├── gold/            # Training data
│   └── models/          # Trained models
├── docs/                # Documentation (local only)
└── feature_repo/        # Feast feature store
```

## Configuration

Required environment variables in `.env`:
```bash
OANDA_TOKEN=your_token_here
OANDA_ACCOUNT_ID=your_account_id
```

## License

For educational and research purposes.

---

**Status**: ✅ Pipeline Ready | ⚠️ Collecting News Data
