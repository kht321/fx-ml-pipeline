# Clean Medallion Architecture - Data Structure

Generated: 2025-10-16 02:00:22

## Directory Structure

```
data_clean/
├── bronze/              # Raw data (immutable)
│   ├── market/         # OHLCV candles from OANDA
│   └── news/           # Raw news articles (JSON)
│
├── silver/             # Processed features
│   ├── market/
│   │   ├── technical/      # RSI, MACD, Bollinger, etc.
│   │   ├── microstructure/ # Spread, volume, depth
│   │   └── volatility/     # GK, Parkinson, YZ estimators
│   └── news/
│       ├── sentiment/      # Sentiment scores
│       ├── entities/       # Entity mentions
│       └── topics/         # Topic classifications
│
├── gold/               # Training-ready data
│   ├── market/
│   │   ├── features/       # Merged market features
│   │   └── labels/         # Price prediction labels
│   └── news/
│       └── signals/        # Trading signals from news
│
├── models/             # Trained models
└── training_outputs/   # Training logs, metrics, plots
```

## Data Flow

### Market Data Pipeline
```
Bronze (NDJSON) → Silver (CSV) → Gold (CSV + Parquet)
  Raw candles   →   Features   →   Training data
```

### News Data Pipeline
```
Bronze (JSON) → Silver (CSV) → Gold (CSV)
 Raw articles →   Features   → Trading signals
```

## Timezone Standards

- **All timestamps in UTC**
- Market data: ISO 8601 format with 'Z' suffix
- News data: ISO 8601 format, normalized to UTC

## File Naming Conventions

### Bronze Layer
- Market: `{instrument}_{granularity}_{years}y_{date}.ndjson`
  Example: `spx500_usd_m1_5y_20251016.ndjson`

- News: `{article_id}.json`
  Example: `a1b2c3d4e5f6.json`

### Silver Layer
- Market: `{instrument}_{feature_type}_{date}.csv`
  Example: `spx500_technical_20251016.csv`

- News: `{source}_{feature_type}_{date}.csv`
  Example: `all_sources_sentiment_20251016.csv`

### Gold Layer
- Market: `{instrument}_features_{date}.csv`
  Example: `spx500_features_20251016.csv`

- Labels: `{instrument}_labels_{horizon}.csv`
  Example: `spx500_labels_30min.csv`

- News: `{instrument}_news_signals_{date}.csv`
  Example: `spx500_news_signals_20251016.csv`

## Data Quality Checks

1. **Timestamp Validation**: All timestamps must be valid UTC
2. **Completeness**: No missing required fields
3. **Deduplication**: Remove duplicate candles/articles
4. **Alignment**: Market and news data must overlap temporally

## Usage

See individual pipeline scripts in `src_clean/data_pipelines/`

