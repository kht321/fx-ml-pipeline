# OANDA SGD FX ML Pipeline

This project implements a **dual medallion architecture** for modeling Singapore dollar (SGD) FX moves with live data from OANDA v20 and financial news sources. The system separates Market Data and News processing into independent pipelines, each following Bronze → Silver → Gold progression, with FinGPT-enhanced sentiment analysis and a final Combined layer for multi-modal modeling.

## 🎯 Project Overview

This pipeline is designed for **real-time SGD FX prediction** using:
- **OANDA Paper Trading Account**: Live USD_SGD price streaming for market data
- **Simulated News Feeds**: Both scraped 2025 news and synthetically generated test data
- **Dual Medallion Architecture**: Parallel processing of market and news pipelines
- **FinGPT Sentiment Analysis**: Financial domain-specific sentiment extraction
- **XGBoost Modeling**: Time-series feature engineering with lag-based predictions

## 🚀 Live Data Collection Strategy

### **Data Sources**

#### **Market Data: OANDA Paper Trading Account**
- **Source**: OANDA v20 API live streaming endpoint
- **Instrument**: USD_SGD (primary), EUR_USD, GBP_USD (context)
- **Format**: **NDJSON (Newline-Delimited JSON)**
- **Frequency**: Hourly candles (H1 granularity)
- **Components**: OHLCV + Bid/Ask spreads + Volume

**Why Paper Trading Account?**
- Provides real market data without financial risk
- Identical API structure to live trading accounts
- Sufficient for ML model development and backtesting
- Full historical data access for feature engineering

#### **News Data: Dual Simulator Approach**
The pipeline uses **two separate news datasets** for training and testing:

1. **Scraped 2025 News** (Real-world data)
   - Sources: Reuters, Bloomberg, Channel News Asia, Straits Times
   - Method: RSS feeds + web scraping
   - Filter: SGD-relevant keywords (MAS, Singapore economy, ASEAN, USD strength)
   - Purpose: Real-world training data with actual market correlations

2. **Synthetically Generated News** (Test data)
   - Method: Template-based generation with controlled scenarios
   - Content: Predefined market events (MAS policy shifts, China stimulus, etc.)
   - Purpose: Reproducible testing, edge case simulation, regression testing
   - Advantages: Known ground truth, controlled timing, consistent formatting

**Why Simulated News vs Real API?**
- Financial news APIs are expensive ($500-2000/month for Bloomberg, Reuters)
- Scraping provides sufficient coverage for SGD-specific content
- Synthetic data enables controlled testing scenarios
- Combination ensures both realism and reproducibility

### **NDJSON Format: Why It's Optimal for Streaming**

All Bronze layer data is stored in **NDJSON (Newline-Delimited JSON)** format:

```json
{"time": "2025-01-15T08:00:00.000000Z", "instrument": "USD_SGD", "close": 1.3452, ...}
{"time": "2025-01-15T09:00:00.000000Z", "instrument": "USD_SGD", "close": 1.3455, ...}
{"time": "2025-01-15T10:00:00.000000Z", "instrument": "USD_SGD", "close": 1.3448, ...}
```

**Design Advantages:**

1. **Append-Only Streaming**
   - Each new data point is appended as a single line
   - No need to rewrite entire file (crucial for live streaming)
   - Perfect for OANDA's continuous price stream

2. **Crash Recovery**
   - If collection stops, last complete line is valid
   - No corrupted JSON from incomplete writes
   - Resume from last timestamp without data loss

3. **Memory Efficiency**
   - Process one line at a time (streaming processing)
   - No need to load entire dataset into memory
   - Scales to years of tick data

4. **Simple Parsing**
   ```python
   # Read line-by-line for feature engineering
   with open('usd_sgd_hourly_2025.ndjson', 'r') as f:
       for line in f:
           candle = json.loads(line)
           # Process incrementally
   ```

5. **Easy Deduplication**
   - Check last line for latest timestamp
   - Fetch only new data from OANDA API
   - No complex state management

**Alternative Formats Considered:**
- ❌ **CSV**: Schema changes break parsers, no nested structures
- ❌ **Single JSON Array**: Requires rewriting entire file on each append
- ❌ **Parquet**: Optimized for batch processing, not streaming appends
- ✅ **NDJSON**: Perfect balance of simplicity, streaming, and robustness

### Quick Start 2025 Collection
```bash
# Install dependencies
pip install -e .

# Start live collection (market + news)
python scripts/start_2025_collection.py

# Check status
python scripts/start_2025_collection.py --status

# Test news collection only
python scripts/start_2025_collection.py --test-news
```

### **Bronze Layer Data Schema**

**Market Data** ([usd_sgd_hourly_2025.ndjson](data/bronze/prices/usd_sgd_hourly_2025.ndjson)):
```json
{
  "time": "2025-01-15T08:00:00.000000Z",
  "instrument": "USD_SGD",
  "granularity": "H1",
  "open": 1.3450,
  "high": 1.3460,
  "low": 1.3445,
  "close": 1.3452,
  "volume": 1234,
  "bid_open": 1.3449, "bid_high": 1.3459, "bid_low": 1.3444, "bid_close": 1.3451,
  "ask_open": 1.3451, "ask_high": 1.3461, "ask_low": 1.3446, "ask_close": 1.3453,
  "spread": 0.0002,
  "collected_at": "2025-01-15T08:05:23.123456Z"
}
```

**News Data** ([financial_news_2025.ndjson](data/bronze/news/financial_news_2025.ndjson)):
```json
{
  "url": "https://www.reuters.com/markets/currencies/...",
  "title": "MAS maintains monetary policy stance amid inflation concerns",
  "content": "The Monetary Authority of Singapore (MAS) announced...",
  "summary": "Singapore's central bank maintains policy...",
  "published": "2025-01-15T07:30:00Z",
  "source": "reuters_singapore",
  "scraped_at": "2025-01-15T08:00:00.000000Z",
  "sgd_relevant": true,
  "word_count": 456,
  "char_count": 2891
}
```

## 🏗️ Architecture Overview

### **Why Dual Medallion Architecture?**

This pipeline implements **two separate medallion pipelines** (Market + News) instead of a single unified pipeline. Here's why:

#### **1. Different Data Velocities**

**Market Data**: High-frequency, continuous streaming
- OANDA streams price ticks every few seconds
- Hourly candles complete every 60 minutes
- **Processing**: Real-time, incremental updates

**News Data**: Low-frequency, event-driven
- Articles published irregularly (0-20 per day)
- Scraping runs every 30 minutes
- **Processing**: Batch processing on arrival

**Problem with Single Pipeline**: Forcing both into the same cadence would either:
- Over-process market data (unnecessary batch reprocessing)
- Under-process news data (missing timely sentiment signals)

#### **2. Independent Scaling Requirements**

**Market Pipeline**:
- Lightweight transformations (technical indicators, rolling stats)
- CPU-bound feature engineering
- Scales horizontally (multiple instruments in parallel)
- **Bottleneck**: I/O for tick storage

**News Pipeline**:
- Heavy NLP processing (FinGPT inference)
- GPU-bound sentiment analysis
- Requires 16GB+ RAM for transformer models
- **Bottleneck**: FinGPT inference latency (~2-5s per article)

**Dual Pipeline Advantage**: Each pipeline can be deployed with appropriate resources:
- Market: Lightweight containers, fast I/O
- News: GPU instances, high memory

#### **3. Separate Failure Domains**

**Isolation Benefits**:
- If FinGPT fails (OOM, GPU issues), market pipeline continues
- If OANDA API has downtime, news scraping continues
- Each pipeline has independent retry logic and error handling

**Real-world Example**:
```
T=10:00: Market pipeline collecting USD_SGD ticks normally
T=10:05: FinGPT crashes due to CUDA OOM
T=10:06: News pipeline falls back to lexicon-based sentiment
T=10:10: Market pipeline continues unaffected, no data loss
T=10:15: FinGPT restarts, news pipeline resumes normal operation
```

#### **4. Data Quality and Lineage**

**Market Bronze → Silver → Gold**:
- Each transformation is deterministic (mathematical calculations)
- Full audit trail: Bronze (raw ticks) → Silver (indicators) → Gold (features)
- Easy to debug: Check Silver CSVs if Gold features look wrong

**News Bronze → Silver → Gold**:
- Non-deterministic transformations (FinGPT sentiment)
- Version control: Track FinGPT model version in metadata
- Reproducibility: Synthetic news for regression testing

**Why Not Merge at Bronze?**
- Mixing raw ticks with raw news articles creates a messy schema
- Hard to version control when one side changes
- Difficult to debug which pipeline has issues

#### **5. Flexible Model Development**

The dual architecture enables **three modeling strategies**:

1. **Market-Only Models**: Low-latency predictions (65-70% accuracy)
   ```python
   model = train(data/market/gold/training/market_features.csv)
   ```

2. **News-Only Models**: Event-driven signals (60-65% accuracy)
   ```python
   model = train(data/news/gold/news_signals/sentiment_features.csv)
   ```

3. **Combined Models**: Best accuracy (78-85% accuracy)
   ```python
   # Merge only during training, not in storage
   market_gold = load('data/market/gold/training/market_features.csv')
   news_silver = load('data/news/silver/sentiment_scores/sentiment_features.csv')
   combined_features = merge_asof(market_gold, news_silver)
   model = train(combined_features)
   ```

**Key Insight**: Gold layers remain **separate in storage**, merged **temporarily during training**. This keeps storage clean and enables independent pipeline optimization.

#### **6. Medallion Pattern Rationale**

**Bronze (Raw)**: Immutable source of truth
- Market: Exact OANDA API response (NDJSON)
- News: Exact scraped content (NDJSON)
- **Never modified** after collection
- Enables full pipeline reprocessing if feature logic changes

**Silver (Features)**: Derived, versioned transformations
- Market: Technical indicators, volatility regimes
- News: FinGPT sentiment scores, entity mentions
- **Can be regenerated** from Bronze if needed
- Stores intermediate features for debugging

**Gold (Training-Ready)**: Model-ready datasets
- Market: Merged Silver CSVs + time features
- News: Aggregated sentiment signals
- **Optimized for ML training**: No missing values, aligned timestamps
- Final quality checks before modeling

**Why Three Layers Instead of Two?**
- Bronze → Gold would mix raw data cleaning with feature engineering
- Silver layer provides a checkpoint for debugging
- Enables different teams to work on Bronze→Silver vs Silver→Gold independently

---

## 🔍 Deep Dive: Feature Engineering Design Rationale

### **Q1: Why 3 Separate CSVs for Market Silver Layer?**

**Short Answer**: Different feature families have different computational dependencies and update frequencies.

#### **The Three CSV Separation Strategy**

```python
# Market Silver Layer Structure
data/market/silver/
├── technical_features/      # Price-based features
│   └── sgd_vs_majors.csv
├── microstructure/          # Liquidity & order book
│   └── depth_features.csv
└── volatility/              # Risk metrics
    └── risk_metrics.csv
```

**Rationale by Category:**

**1. Technical Features** (technical_features/sgd_vs_majors.csv)
- **Source**: Derived from Bronze `mid` price and `spread`
- **Dependencies**: Only needs price time series
- **Update Frequency**: Every tick (real-time streaming)
- **Example Features**:
  ```python
  ret_1 = mid.pct_change(1)           # 1-tick return
  roll_vol_20 = ret_1.rolling(20).std()  # Rolling volatility
  ewma_short = mid.ewm(span=5).mean()    # Exponential MA
  zscore_20 = (mid - roll_mean_20) / roll_vol_20  # Price deviation
  ```
- **Why Separate**: Pure mathematical transformations, no external dependencies

**2. Microstructure Features** (microstructure/depth_features.csv)
- **Source**: Derived from Bronze `bid_liquidity`, `ask_liquidity`, `spread`
- **Dependencies**: Requires order book data (not always available)
- **Update Frequency**: Only when order book snapshots are fetched
- **Example Features**:
  ```python
  total_liquidity = bid_liquidity + ask_liquidity
  liquidity_imbalance = (ask_liquidity - bid_liquidity) / total_liquidity
  effective_spread = spread / mid  # Trading cost metric
  ```
- **Why Separate**: Order book data is sparse (not in every tick), would create gaps in technical features

**3. Volatility Features** (volatility/risk_metrics.csv)
- **Source**: Derived from Bronze `mid` and computed `ret_1`
- **Dependencies**: Needs sufficient historical data (60+ ticks for long windows)
- **Update Frequency**: Every tick, but features mature slowly
- **Example Features**:
  ```python
  vol_5 = ret_1.rolling(5).std() * sqrt(5)      # 5-tick volatility
  vol_60 = ret_1.rolling(60).std() * sqrt(60)   # 60-tick volatility
  vol_percentile = vol_20.rolling(100).rank(pct=True)  # Regime detection
  high_vol_regime = (vol_percentile > 0.8).astype(int)  # Binary flag
  ```
- **Why Separate**: Different window sizes (5 vs 60 vs 100), some features need long lookback

#### **Key Advantages of 3-CSV Separation**

| Scenario | Benefit |
|----------|---------|
| **Order book API fails** | Microstructure CSV will be empty, but technical & volatility features continue |
| **Debugging volatility spikes** | Can examine `volatility/risk_metrics.csv` in isolation without noise from other features |
| **Feature selection** | Can easily drop entire microstructure category if model doesn't need it |
| **Independent updates** | Technical features update every tick, microstructure only on order book snapshots |
| **Storage efficiency** | Microstructure CSV is much smaller (only when order book data available) |

**Alternative (Single CSV)**: Would create sparse matrix with many NaN values when order book unavailable
**Our Choice (3 CSVs)**: Dense matrices per feature family, clean joins at Gold layer

---

### **Q2: Why Are Some Features in Silver vs Others in Gold?**

**Critical Distinction**:
- **Silver = Stateless transformations** (single-instrument, no cross-dependencies)
- **Gold = Stateful aggregations** (cross-instrument, time-aware features)

#### **Silver Layer (Stateless Features)**

Features that can be computed **per instrument independently**:

```python
# Technical features - only need USD_SGD data
ret_1 = usd_sgd['mid'].pct_change(1)
vol_20 = usd_sgd['ret_1'].rolling(20).std()
zscore_20 = (usd_sgd['mid'] - usd_sgd['mid'].rolling(20).mean()) / vol_20

# Microstructure - only need USD_SGD order book
liquidity_imbalance = (usd_sgd['ask_liquidity'] - usd_sgd['bid_liquidity']) / total_liquidity

# Volatility - only need USD_SGD returns
vol_percentile = usd_sgd['vol_20'].rolling(100).rank(pct=True)
```

**Why in Silver**: Each instrument processes independently, no cross-dependencies

#### **Gold Layer (Stateful Features)**

Features that require **multiple instruments** or **cross-time aggregations**:

```python
# Cross-instrument correlation (needs EUR_USD + USD_SGD)
usd_sgd_eur_usd_corr = rolling_corr(
    usd_sgd['ret_1'],
    eur_usd['ret_1'],
    window=50
)

# Relative performance (needs all majors)
major_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
basket_return = mean([pair['ret_1'] for pair in major_pairs])
relative_performance = usd_sgd['ret_1'] - basket_return

# Time features (needs current timestamp context)
hour = usd_sgd['time'].dt.hour
asian_session = (hour >= 0) & (hour < 9)  # 0-9 AM SGT
london_session = (hour >= 15) & (hour < 24)  # 3 PM - midnight SGT
```

**Why in Gold**: Requires joining multiple Silver CSVs + time-zone aware logic

#### **The Layering Logic**

```
Bronze (Raw) → "What OANDA sent us"
  ↓
Silver (Stateless) → "What we can compute per instrument"
  - Technical: f(USD_SGD prices only)
  - Microstructure: f(USD_SGD order book only)
  - Volatility: f(USD_SGD returns only)
  ↓
Gold (Stateful) → "What we need multiple instruments for"
  - Cross-correlations: f(USD_SGD + EUR_USD)
  - Relative performance: f(USD_SGD + all majors)
  - Time features: f(current_time + trading_schedule)
```

**Design Principle**:
- Keep Silver **horizontally scalable** (process instruments in parallel)
- Reserve Gold for **vertical integration** (merge outputs, add cross-dependencies)

---

For complete technical details, architecture diagrams, and implementation guides, see the full documentation in [docs/](docs/).
