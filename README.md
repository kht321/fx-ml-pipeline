# OANDA FX ML Pipeline

This repository streams FX prices from the OANDA v20 API, engineers basic features, and trains a baseline logistic regression model for price direction classification. The pipeline is intentionally lightweight so you can run experiments quickly against a practice account.

## Project Layout

```
oanda-fx-ml/
├── .env.example          # Sample environment variables for OANDA credentials
├── configs/
│   ├── features.yaml     # Feature definitions & labeling horizon
│   └── pairs.yaml        # Instrument groupings
├── data/
│   ├── proc/             # Engineered datasets
│   └── raw/              # Raw API payloads
├── src/
│   ├── build_features.py # Transform streamed ticks into a feature matrix
│   ├── fetch_candles.py  # Pull historical candles via REST
│   ├── fetch_orderbook.py# Fetch order book snapshots
│   ├── oanda_api.py      # Thin wrapper around OANDA SDK endpoints
│   ├── stream_prices.py  # Stream live prices and emit JSON lines
│   └── train_baseline.py # Train/test a logistic regression baseline
├── tests/
│   └── __init__.py
├── pyproject.toml        # Project metadata and dependencies
└── README.md
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
   # edit .env with your OANDA token, account id, and environment (practice/live)
   export $(grep -v '^#' .env | xargs)
   ```

3. **Stream prices and build features**
   ```bash
   python src/stream_prices.py EUR_USD GBP_USD \
     | python src/build_features.py \
     > data/proc/eu_gb_features.csv
   ```

4. **Fetch historical data**
   ```bash
   python src/fetch_candles.py EUR_USD M1 500 > data/raw/eurusd_m1.json
   python src/fetch_orderbook.py EUR_USD > data/raw/eurusd_orderbook.json
   ```

5. **Train the baseline model**
   ```bash
   python src/train_baseline.py data/proc/eu_gb_features.csv
   ```
   The script prints a JSON summary that includes the classification report and model coefficients.

## Notes

- Streaming uses the `PricingStream` endpoint (max 4 price updates/sec per instrument). Heartbeat messages are ignored in `stream_prices.py`.
- Candle fetches use the `InstrumentsCandles` endpoint with price types `M`, `B`, and `A` for midpoint/bid/ask.
- Order book snapshots come from `InstrumentsOrderBook`.
- The feature builder expects newline-delimited price ticks from `stream_prices.py` on `stdin`.
- Update `configs/` to manage the instruments tracked and feature engineering assumptions.
