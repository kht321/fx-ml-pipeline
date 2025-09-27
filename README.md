# OANDA SGD FX ML Pipeline

This project implements a medallion-style workflow for modelling Singapore dollar FX moves with data pulled from OANDA v20. The Bronze layer captures raw price ticks, candles, and curated news drops; the Silver layer engineers aligned features; the Gold layer consolidates everything into trainable datasets and models.

## Project Layout

```
oanda-fx-ml/
├── .env.example
├── configs/
│   ├── features.yaml
│   └── pairs.yaml
├── data/
│   ├── bronze/
│   │   ├── news/
│   │   ├── orderbook/
│   │   └── prices/
│   ├── silver/
│   │   ├── news/
│   │   └── prices/
│   └── gold/
│       ├── models/
│       └── training/
├── src/
│   ├── build_features.py
│   ├── build_training_set.py
│   ├── fetch_candles.py
│   ├── fetch_orderbook.py
│   ├── oanda_api.py
│   ├── process_news.py
│   ├── stream_prices.py
│   └── train_baseline.py
└── tests/
    └── __init__.py
```

## Getting Started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Configure credentials**
   ```bash
   cp .env.example .env
   # Fill in OANDA_TOKEN, OANDA_ACCOUNT_ID, and OANDA_ENV (practice/live)
   export $(grep -v '^#' .env | xargs)
   ```

3. **Bronze layer – capture raw data**
   ```bash
   # Stream live prices and archive raw ticks
   python src/stream_prices.py USD_SGD EUR_USD GBP_USD \
     --bronze-path data/bronze/prices/usd_sgd_stream.ndjson

   # Fetch historical candles
   python src/fetch_candles.py USD_SGD --granularity M1 --count 2000 \
     --output data/bronze/prices/usdsgd_m1.json
   python src/fetch_candles.py EUR_USD --granularity M1 --count 2000 \
     --output data/bronze/prices/eurusd_m1.json

   # Snapshot supported order books (majors only)
   python src/fetch_orderbook.py EUR_USD \
     --output data/bronze/orderbook/eurusd_orderbook.json
   ```
   Drop curated news stories (text or JSON) into `data/bronze/news/`. They are treated as the raw Bronze feed for macro/LLM features.

4. **Silver layer – engineer features**
   ```bash
   # Promote price ticks to engineered features
   python src/build_features.py \
     --input data/bronze/prices/usd_sgd_stream.ndjson \
     --output data/silver/prices/sgd_vs_majors.csv

   # Convert curated news into numeric sentiment features
   python src/process_news.py \
     --input-dir data/bronze/news \
     --silver-path data/silver/news/news_features.csv
   ```
   `process_news.py --follow` can run as a lightweight watcher that ingests new files as they arrive.

5. **Gold layer – align price and news signals**
   ```bash
   python src/build_training_set.py \
     --price-features data/silver/prices/sgd_vs_majors.csv \
     --news-features data/silver/news/news_features.csv \
     --output data/gold/training/sgd_vs_majors_training.csv
   ```
   The script performs an as-of join so that each price observation carries the latest relevant news context within a configurable lookback window (default 6h).

6. **Train the predictive baseline**
   ```bash
   python src/train_baseline.py \
     data/gold/training/sgd_vs_majors_training.csv \
     --model-output data/gold/models/logreg_baseline.pkl
   ```
   The command prints a classification report and stores the fitted scaler + model bundle for downstream use.

## News Ingestion Notes

- `src/process_news.py` supports plain-text or JSON files. JSON documents can expose keys such as `headline`, `body`, `published_at`, and `source`; missing metadata falls back to file timestamps.
- Feature extraction currently uses a lightweight lexicon for sentiment and tags Singapore-related terms. Replace the heuristics or extend the script to call your preferred LLM service when you are ready for richer embeddings.
- Processed rows append to `data/silver/news/news_features.csv` and are tracked via `data/bronze/news/.processed.json` to prevent duplicate ingestion.

## Extending the Pipeline

- Adjust `configs/pairs.yaml` to switch instrument baskets, and `configs/features.yaml` to document newly engineered factors.
- Update `build_training_set.py --news-tolerance` to widen or narrow the lookback window for associating news with ticks.
- Swap `train_baseline.py` for more advanced models (gradient boosting, sequence models) once the Gold dataset is stable. The script saves the StandardScaler alongside the classifier to ease deployment.

## OANDA API References

- Streaming: `PricingStream` (max 4 price updates/sec per instrument). Heartbeats can be persisted via `--include-heartbeats` on `stream_prices.py`.
- Historical data: `InstrumentsCandles` with configurable price components (`--price` accepts combinations such as `M`, `B`, `A`).
- Order book snapshots: `InstrumentsOrderBook` is available for major pairs (e.g. `EUR_USD`, `GBP_USD`, `USD_JPY`).

With these building blocks you can automate Bronze capture, incrementally refresh Silver features, and keep the Gold training table up to date for SGD-centric FX modelling.
