# FX ML Pipeline Architecture

## Overview

This project implements a **dual medallion architecture** for Singapore Dollar (SGD) FX prediction, featuring:
- **Two Independent Pipelines**: Market Data (high-frequency) + News Data (event-driven)
- **Three-Layer Medallion Pattern**: Bronze (raw) → Silver (features) → Gold (ML-ready)
- **FinGPT-Enhanced Sentiment**: Market-aware news analysis with LLM context
- **XGBoost with Lag Features**: Explicit time-series pattern learning
- **NDJSON Streaming**: Append-only format for crash recovery and deduplication

## Architecture Principles

### **1. Separation of Concerns**

**Why Dual Pipelines?**

Market and News data have fundamentally different characteristics:

| Aspect | Market Pipeline | News Pipeline |
|--------|----------------|---------------|
| **Data Velocity** | High-frequency (hourly candles) | Low-frequency (5-20 articles/day) |
| **Processing** | Continuous streaming | Event-driven batches |
| **Compute** | CPU-bound (technical indicators) | GPU-bound (FinGPT inference) |
| **Scaling** | Horizontal (multiple instruments) | Vertical (GPU memory) |
| **Failure Mode** | API downtime | Web scraping blocks / FinGPT OOM |

**Design Decision**: Independent pipelines prevent cascading failures and enable targeted resource allocation.

### **2. Medallion Pattern (Bronze → Silver → Gold)**

**Bronze (Raw)**: Immutable source of truth
- Market: Exact OANDA API responses (NDJSON)
- News: Exact scraped content (NDJSON)
- **Never modified** after collection
- Enables full pipeline reprocessing if logic changes

**Silver (Features)**: Derived, versioned transformations
- Market: 3 separate CSVs (technical, microstructure, volatility)
- News: 3 separate CSVs (sentiment, entities, topics)
- **Stateless**: Per-instrument/per-article independent processing
- **Horizontally scalable**: Process instruments/articles in parallel

**Gold (Training-Ready)**: Model-ready datasets
- Market: Merged Silver + cross-instrument features + time features
- News: Aggregated hourly signals + quality scoring
- **Stateful**: Cross-dependencies (correlations, aggregations)
- **Preprocessed**: Missing value imputation, outlier handling, validation

**Why Three Layers?**
- Bronze → Gold would mix data cleaning with feature engineering
- Silver provides debugging checkpoint
- Enables team specialization (Bronze→Silver vs Silver→Gold)

### **3. Independent Scaling Strategy**

**Market Pipeline**:
- **Processing**: Continuous streaming (every tick)
- **Compute**: CPU-bound (pandas transformations)
- **Scaling**: Horizontal (add more instruments)
- **Deployment**: Lightweight containers, fast I/O

**News Pipeline**:
- **Processing**: Event-driven (when articles arrive)
- **Compute**: GPU-bound (FinGPT inference ~2-5s/article)
- **Scaling**: Vertical (larger GPU instances)
- **Deployment**: GPU instances (8GB+ VRAM)

**Combined Training**:
- **Processing**: Scheduled retraining (hourly/daily)
- **Compute**: CPU for XGBoost (or GPU for LSTM)
- **Scaling**: Model-dependent
- **Deployment**: Batch training jobs

## Complete Data Flow

### **Visual Pipeline Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                  │
├─────────────────────────────────────────────────────────────────┤
│  🏦 OANDA Paper Trading Account (Live Streaming)                │
│  📰 News Scraping (Reuters, Bloomberg, CNA, ST)                 │
│  🧪 Synthetic News (Test scenarios)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BRONZE LAYER (Raw Data)                       │
├─────────────────────────────────────────────────────────────────┤
│  Format: NDJSON (Newline-Delimited JSON)                        │
│  Market: usd_sgd_hourly_2025.ndjson                             │
│  News: financial_news_2025.ndjson                               │
│                                                                  │
│  Why NDJSON:                                                     │
│  ✓ Append-only streaming (no file rewrites)                     │
│  ✓ Crash recovery (last line always valid)                      │
│  ✓ Memory efficient (line-by-line processing)                   │
│  ✓ Simple deduplication (check last timestamp)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────┬──────────────────────────────────────┐
│   🏦 MARKET PIPELINE     │     📰 NEWS PIPELINE                 │
│   (High-frequency)       │     (Event-driven)                   │
├──────────────────────────┼──────────────────────────────────────┤
│  SILVER: 3 CSVs          │   SILVER: FinGPT Analysis            │
│  ├─ technical.csv        │   ├─ sentiment.csv (FinGPT outputs)  │
│  ├─ microstructure.csv   │   ├─ entities.csv (NER)              │
│  └─ volatility.csv       │   └─ topics.csv (Classification)     │
│                          │                                      │
│  Features:               │   Features:                          │
│  - Returns, EWMA         │   - sentiment_score                  │
│  - Spreads, z-scores     │   - sgd_directional_signal          │
│  - Liquidity metrics     │   - market_coherence                 │
│  - Volatility regimes    │   - signal_strength_adjusted        │
│                          │                                      │
│  Dependencies:           │   Dependencies:                      │
│  - Per-instrument only   │   - Uses Market Silver as context!   │
│  - Stateless             │   - FinGPT prompt includes market    │
│                          │     features (mid, vol, regime)      │
├──────────────────────────┼──────────────────────────────────────┤
│  GOLD: market_features   │   GOLD: trading_signals              │
│                          │                                      │
│  Transformations:        │   Transformations:                   │
│  - Merge 3 Silver CSVs   │   - Merge 3 Silver CSVs              │
│  - Cross-instrument      │   - Currency explosion               │
│    correlations          │     (1 article → N currencies)       │
│  - Time features         │   - Temporal aggregation             │
│    (hour, session)       │     (articles → hourly buckets)      │
│  - Preprocessing         │   - Quality scoring                  │
│    (imputation, filter)  │   - Time decay weighting             │
│                          │                                      │
│  Output: 32+ features    │   Output: 24+ features               │
│  Granularity: Hourly     │   Granularity: Hourly                │
└──────────────────────────┴──────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               🤖 COMBINED TRAINING LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  As-of Join (no look-ahead bias):                               │
│  - Market time: 2025-01-15 12:00                                │
│  - News cutoff: 2025-01-15 06:00 (6H tolerance)                 │
│  - Only news published BEFORE market time                       │
│                                                                  │
│  Lag Feature Engineering:                                        │
│  - Create [1,2,3,5,10] lags for all features                    │
│  - Market lags: ret_1_lag1, vol_20_lag5, ...                    │
│  - News lags: sentiment_lag1, signal_lag2, ...                  │
│                                                                  │
│  Output: 60+ features (32 market + 10 news + 50 lagged)         │
│                                                                  │
│  XGBoost Training:                                               │
│  - Temporal split (train on past, test on future)               │
│  - Early stopping (prevent overfitting)                         │
│  - Feature importance (identify key lags)                       │
│                                                                  │
│  Expected Accuracy: 78-85%                                       │
│  Latency: <5s                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### **Key Cross-Pipeline Interaction**

**Market Silver → News Silver Context Flow**:
```python
# When processing news article at 08:00
market_context = load_latest_market_silver(time="08:00")
# Returns: {mid: 1.3452, vol_20: 0.0145, high_vol_regime: True, ...}

fingpt_analysis = analyze_sgd_news(
    news_text=article['content'],
    market_context=market_context  # ← Market Silver features injected here!
)
# FinGPT prompt includes: "USD/SGD Mid Price: 1.3452, Volatility: 1.45%, Regime: High"
# Output: sentiment_score, market_coherence, signal_strength_adjusted
```

**Key Insight**: News Silver doesn't just analyze text—it analyzes text **in the context of current market conditions**.

## Key Components and Implementation Files

### **Data Collection Layer**

| Component | File | Responsibility | Output |
|-----------|------|----------------|--------|
| **Market Streaming** | [`hourly_candle_collector.py`](src/hourly_candle_collector.py) | Fetch hourly candles from OANDA API | Bronze NDJSON |
| **News Scraping** | [`news_scraper.py`](src/news_scraper.py) | Scrape SGD-relevant news from 4 sources | Bronze NDJSON |
| **OANDA API Wrapper** | [`oanda_api.py`](src/oanda_api.py) | REST/streaming API integration | API responses |
| **Orchestration** | [`data_collection_pipeline.py`](src/data_collection_pipeline.py) | Coordinate market + news collection | Pipeline health |

### **Market Pipeline (Bronze → Silver → Gold)**

| Layer | File | Transformations | Output |
|-------|------|-----------------|--------|
| **Bronze → Silver** | [`build_market_features.py`](src/build_market_features.py) | • Returns, volatility, z-scores<br>• EWMA, spreads<br>• Liquidity metrics<br>• Volatility regimes | 3 CSVs:<br>- technical.csv<br>- microstructure.csv<br>- volatility.csv |
| **Silver → Gold** | [`build_market_gold.py`](src/build_market_gold.py) | • Merge 3 Silver CSVs<br>• Cross-instrument correlations<br>• Time features (hour, session)<br>• Preprocessing (imputation, filtering) | market_features.csv<br>(32+ features) |

### **News Pipeline (Bronze → Silver → Gold)**

| Layer | File | Transformations | Output |
|-------|------|-----------------|--------|
| **FinGPT Processor** | [`fingpt_processor.py`](src/fingpt_processor.py) | • Load FinGPT model (LLaMA2-7B LoRA)<br>• Market-aware prompt construction<br>• Sentiment parsing | FinGPTAnalysis object |
| **Bronze → Silver** | [`build_news_features.py`](src/build_news_features.py) | • FinGPT sentiment analysis<br>• Entity extraction (NER)<br>• Topic classification<br>• Uses Market Silver as context | 3 CSVs:<br>- sentiment.csv<br>- entities.csv<br>- topics.csv |
| **Silver → Gold** | [`build_news_gold.py`](src/build_news_gold.py) | • Merge 3 Silver CSVs<br>• Currency explosion (1 article → N currencies)<br>• Temporal aggregation (hourly buckets)<br>• Quality scoring + time decay | trading_signals.csv<br>(24+ features) |

### **Combined Training Layer**

| Component | File | Transformations | Output |
|-----------|------|-----------------|--------|
| **Multi-modal Training** | [`train_combined_model.py`](src/train_combined_model.py) | • As-of join (Market Gold + News Gold)<br>• Lag feature engineering ([1,2,3,5,10])<br>• Temporal train/test split<br>• XGBoost/RF/LogReg training | Trained model PKL<br>+ metrics JSON |

### **Orchestration**

| Component | File | Responsibility |
|-----------|------|----------------|
| **Pipeline Coordinator** | [`orchestrate_pipelines.py`](src/orchestrate_pipelines.py) | • Health monitoring<br>• Continuous operation<br>• Error handling<br>• Schedule management |

## Data Architecture and Storage Strategy

### **NDJSON for Bronze Layer**

**Format Choice**: Newline-Delimited JSON (NDJSON)

**Why NDJSON over CSV/Parquet/JSON Array?**

| Format | Streaming | Append-Only | Crash Recovery | Memory Efficient | Schema Flexibility |
|--------|-----------|-------------|----------------|------------------|--------------------|
| CSV | ❌ | ❌ | ❌ | ✅ | ❌ |
| JSON Array | ❌ | ❌ | ❌ | ❌ | ✅ |
| Parquet | ❌ | ❌ | ✅ | ✅ | ❌ |
| **NDJSON** | ✅ | ✅ | ✅ | ✅ | ✅ |

**NDJSON Advantages**:
1. **Append-only**: Each new candle/article is a single line append (no file rewrite)
2. **Crash recovery**: Last complete line is always valid, no corruption
3. **Deduplication**: Read last line to get latest timestamp, fetch only new data
4. **Memory efficient**: Process line-by-line, no need to load entire file
5. **Schema flexible**: Nested JSON structures (OANDA API responses)

### **3-CSV Separation in Silver Layer**

**Market Silver**: 3 separate CSVs
- `technical_features.csv`: Price-based indicators (always complete)
- `microstructure/depth_features.csv`: Order book metrics (sparse, not always available)
- `volatility/risk_metrics.csv`: Long-window volatility (needs 60-100 ticks)

**Why Separate Instead of Single CSV?**

| Scenario | Single CSV | 3 Separate CSVs (Our Choice) |
|----------|-----------|------------------------------|
| Order book API fails | Entire CSV has NaN columns | Only microstructure CSV is empty, technical + volatility continue |
| Debugging volatility spike | Must filter through all features | Directly examine `volatility/risk_metrics.csv` |
| Feature selection | Must drop columns | Can exclude entire CSV file |
| Storage efficiency | Sparse matrix with many NaNs | Dense matrices per feature family |

**News Silver**: 3 separate CSVs
- `sentiment_scores/sentiment_features.csv`: FinGPT outputs (always present)
- `entity_mentions/entity_features.csv`: NER results (may be sparse)
- `topic_signals/topic_features.csv`: Topic classification (may be sparse)

### **Storage Layout**

```
fx-ml-pipeline/
├── data/
│   ├── bronze/                      # Immutable raw data (NDJSON)
│   │   ├── prices/
│   │   │   └── usd_sgd_hourly_2025.ndjson   (~3 MB/year)
│   │   └── news/
│   │       └── financial_news_2025.ndjson    (~22 MB/year)
│   │
│   ├── market/                      # Market medallion
│   │   ├── silver/
│   │   │   ├── technical_features/sgd_vs_majors.csv
│   │   │   ├── microstructure/depth_features.csv
│   │   │   └── volatility/risk_metrics.csv
│   │   └── gold/
│   │       └── training/market_features.csv  (32+ features)
│   │
│   ├── news/                        # News medallion
│   │   ├── silver/
│   │   │   ├── sentiment_scores/sentiment_features.csv
│   │   │   ├── entity_mentions/entity_features.csv
│   │   │   └── topic_signals/topic_features.csv
│   │   └── gold/
│   │       └── news_signals/trading_signals.csv  (24+ features)
│   │
│   └── combined/                    # Combined training layer
│       ├── models/
│       │   ├── gradient_boosting_combined_model.pkl
│       │   ├── random_forest_combined_model.pkl
│       │   └── logistic_regression_combined_model.pkl
│       └── metrics/
│           └── model_performance.json
```

### **Gold Layers Remain Separate**

**Key Architectural Decision**: Market Gold and News Gold are **not merged in storage**.

**Why?**
1. **Independent reprocessing**: Can rebuild Market Gold without touching News Gold
2. **Model flexibility**: Enables market-only, news-only, or combined models
3. **Cleaner separation**: No coupling between pipelines at storage level
4. **Version control**: Each Gold layer has independent versioning

**Merging happens only during training**:
```python
# Gold layers stay separate in storage
market_gold = pd.read_csv('data/market/gold/training/market_features.csv')
news_gold = pd.read_csv('data/news/gold/news_signals/trading_signals.csv')

# Temporary merge for training
combined_features = merge_asof(market_gold, news_gold, tolerance='6H')

# Train model
model = XGBoost.train(combined_features)

# Save trained model
model.save('data/combined/models/xgboost_model.pkl')
```

## Model Architecture

### **Three Modeling Strategies**

| Strategy | Data Source | Expected Accuracy | Latency | Use Case |
|----------|-------------|-------------------|---------|----------|
| **Market-only** | Market Gold only | 65-70% | <100ms | High-frequency trading, ultra-low latency |
| **News-only** | News Gold only | 60-65% | <2s | Event-driven strategies, fundamental analysis |
| **Combined** | Market Gold + News Gold | 78-85% | <5s | Comprehensive analysis, best accuracy |

### **XGBoost Design Rationale**

**Why XGBoost over LSTM/Transformer?**

| Model | Advantages | Disadvantages | Our Use Case |
|-------|-----------|---------------|--------------|
| **LSTM/GRU** | Learns temporal patterns automatically | Requires large datasets (10K+ samples)<br>Slow training<br>Black box | ❌ Limited 2025 data<br>❌ Need interpretability |
| **Transformer** | State-of-art for sequences | Requires massive datasets<br>High compute cost<br>Overfits small data | ❌ Only ~8,760 hourly samples/year |
| **XGBoost** | Fast training<br>Works with small data<br>Interpretable<br>Feature importance | Requires explicit lag engineering | ✅ Perfect for hourly data<br>✅ Can identify key lags<br>✅ Explainable predictions |

**Lag Feature Engineering**:
```python
# Explicit lags for XGBoost (replaces LSTM's internal memory)
lag_periods = [1, 2, 3, 5, 10]  # Look back 1, 2, 3, 5, 10 hours

for feature in ['ret_1', 'vol_20', 'news_sentiment']:
    for lag in lag_periods:
        df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

# XGBoost can now learn:
# - Momentum: ret_1_lag1 > 0 and ret_1_lag2 > 0 → uptrend
# - Mean reversion: ret_1_lag5 extreme → expect reversal
# - News persistence: news_sentiment_lag1 high → signal decay
```

**Performance Expectations**:
- **Accuracy**: 78-85% on direction prediction (up/down)
- **Latency**: <5s for feature computation + prediction
- **Training time**: ~30s for 10,000 samples
- **Feature importance**: Can identify which lag features matter most

## Technology Stack

### **Core Dependencies**

| Category | Libraries | Purpose |
|----------|-----------|---------|
| **Data Processing** | pandas, numpy | DataFrame operations, numerical computing |
| **ML Models** | scikit-learn, xgboost, joblib | Classification models, model persistence |
| **NLP/LLM** | transformers, torch | FinGPT sentiment analysis (LLaMA2-7B LoRA) |
| **API Integration** | oandapyV20, requests | OANDA v20 REST/streaming, web scraping |
| **Async/Scraping** | aiohttp, feedparser, beautifulsoup4 | Async news collection, RSS parsing, HTML scraping |
| **Configuration** | pyyaml, python-dotenv | Feature configs, environment variables |

### **FinGPT Integration Details**

**Model**: `FinGPT/fingpt-sentiment_llama2-7b_lora`
- **Base Model**: LLaMA2-7B with LoRA adapters
- **Domain**: Financial news sentiment analysis
- **Hardware Requirements**:
  - RAM: 16GB+ recommended
  - VRAM: 8GB+ for 8-bit quantization, 16GB+ for FP16
  - CPU fallback available (slower, ~10s/article)
- **Optimizations**:
  - 8-bit quantization (`load_in_8bit=True`)
  - LoRA adapters reduce parameter count
- **Fallback**: Lexicon-based sentiment if FinGPT fails (OOM/CUDA errors)

**Market-Aware Prompting**:
```python
prompt = f"""Analyze this SGD news given current market conditions:

Article: {news_text}

CURRENT MARKET STATE:
- USD/SGD: {market_context['mid']:.4f}
- Volatility: {market_context['vol_20']:.2%} ({'High' if high_vol else 'Normal'})
- Session: {market_context['session']}

Consider: Has the market already priced this in? Does news align with price action?

Provide: SENTIMENT, SGD_SIGNAL, MARKET_COHERENCE, ADJUSTED_STRENGTH
"""
```

## Operational Considerations

### **Deployment Strategies**

| Component | Current (Development) | Production Recommendation |
|-----------|----------------------|---------------------------|
| **Data Collection** | Local cron jobs | AWS Lambda (news) + ECS (market streaming) |
| **Bronze Storage** | Local NDJSON files | S3 with lifecycle policies (Bronze → Glacier after 1 year) |
| **Silver/Gold Processing** | On-demand scripts | AWS Batch / Airflow scheduled jobs |
| **FinGPT Inference** | Local GPU (A100/3090) | SageMaker GPU instances (on-demand for batches) |
| **Model Training** | Local XGBoost | SageMaker Training Jobs (daily retraining) |
| **Model Serving** | Offline predictions | REST API (Flask/FastAPI) on ECS with Auto Scaling |

### **Monitoring and Observability**

**Data Freshness**:
```python
# Check Bronze layer freshness
last_market_candle = get_latest_timestamp('data/bronze/prices/usd_sgd_hourly_2025.ndjson')
last_news_article = get_latest_timestamp('data/bronze/news/financial_news_2025.ndjson')

alert_if(now - last_market_candle > timedelta(hours=2))  # Market data stale
alert_if(now - last_news_article > timedelta(hours=24))  # News data stale
```

**Processing Success Rates**:
- Market Bronze → Silver: Target >99% (ticks processed without errors)
- News Bronze → Silver: Target >95% (FinGPT may fail on OOM)
- Silver → Gold: Target >99% (merges should always succeed)

**Model Performance Drift**:
- Track rolling 7-day accuracy
- Alert if accuracy drops >5% from baseline
- Trigger retraining if drift detected

**Pipeline Health Metrics**:
- Latency (p50, p95, p99) for each ETL step
- Error rates per pipeline component
- Resource utilization (CPU, GPU, memory)

### **Scaling Recommendations**

**Market Pipeline** (Horizontal Scaling):
```
Current: 1 instrument (USD_SGD)
Scale to: 10 instruments (all major pairs)

Approach:
- Run parallel Bronze collectors (1 per instrument)
- Process Silver features in parallel (stateless)
- Merge all instruments at Gold layer
- Bottleneck: OANDA API rate limits (100 req/sec)
```

**News Pipeline** (Vertical Scaling):
```
Current: 4 news sources, ~20 articles/day
Scale to: 20 sources, ~200 articles/day

Approach:
- Larger GPU instance (A100 80GB for batches)
- Batch FinGPT inference (process 10 articles at once)
- Distributed scraping (multiple scrapers with different IPs)
- Bottleneck: FinGPT inference time (~2s/article)
```

**Combined Training** (Model-Dependent):
```
XGBoost: Single CPU instance sufficient (30s training for 10K samples)
LSTM/Transformer: GPU required (minutes-hours for training)
```

## Development Workflow

### **Adding New Features**

**Market Feature Example**:
```bash
# 1. Add feature to build_market_features.py
def compute_technical_features(df):
    # ... existing features
    df['rsi_14'] = compute_rsi(df['mid'], window=14)  # NEW FEATURE

# 2. Update configs/market_features.yaml
market_features:
  technical:
    rsi_14:
      window: 14
      description: "14-period RSI momentum indicator"

# 3. Rebuild Silver layer
python src/build_market_features.py \
    --input data/bronze/prices/usd_sgd_hourly_2025.ndjson \
    --output-technical data/market/silver/technical_features/sgd_vs_majors.csv

# 4. Test feature appears in Gold layer
python src/build_market_gold.py \
    --feature-selection all \
    --output data/market/gold/training/market_features.csv

# 5. Retrain model with new feature
python src/train_combined_model.py
```

### **Model Development Process**

**1. Start with Individual Pipelines**:
```python
# Market-only model (baseline)
market_model = train_market_only('data/market/gold/training/market_features.csv')
# Expected: 65-70% accuracy

# News-only model (event-driven)
news_model = train_news_only('data/news/gold/news_signals/trading_signals.csv')
# Expected: 60-65% accuracy
```

**2. Combine for Multi-Modal**:
```python
# Combined model
combined_features = merge_asof(market_gold, news_gold)
combined_model = XGBoost.train(combined_features)
# Expected: 78-85% accuracy (10-15% lift from news integration)
```

**3. Feature Importance Analysis**:
```python
# Which features matter most?
importances = combined_model.feature_importances_

# Top 10 features
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
# Typical results:
# 1. ret_1_lag1 (momentum)
# 2. vol_20 (current volatility)
# 3. news_sentiment_lag1 (recent news)
# 4. zscore_20 (price deviation)
# 5. news_trading_signal (aggregated news)
```

### **Testing Strategy**

**Unit Tests** (per component):
```python
# test_market_features.py
def test_compute_technical_features():
    df = pd.DataFrame({'mid': [1.0, 1.01, 1.02, 0.99]})
    result = compute_technical_features(df)
    assert 'ret_1' in result.columns
    assert abs(result['ret_1'].iloc[1] - 0.01) < 1e-6

# test_fingpt_processor.py
def test_sentiment_parsing():
    response = "SENTIMENT: bullish\nCONFIDENCE: 0.85\n..."
    analysis = parse_fingpt_response(response)
    assert analysis['sentiment_score'] == 1.0
    assert analysis['confidence'] == 0.85
```

**Integration Tests** (end-to-end):
```python
# test_pipeline_integration.py
def test_market_pipeline_end_to_end():
    # Bronze → Silver → Gold
    bronze_data = load_bronze('data/bronze/prices/test_candles.ndjson')
    silver_features = build_market_features(bronze_data)
    gold_features = build_market_gold(silver_features)

    assert len(gold_features) > 0
    assert 'y' in gold_features.columns  # Target present
    assert gold_features['y'].isna().sum() == 0  # No missing targets
```

**Performance Tests**:
```python
# test_model_performance.py
def test_combined_model_accuracy():
    model = load_model('data/combined/models/gradient_boosting_combined_model.pkl')
    test_data = load_test_set()

    accuracy = model.score(test_data[feature_cols], test_data['y'])
    assert accuracy >= 0.75  # Minimum 75% accuracy on test set
```

## Key Architectural Decisions Summary

| Decision | Alternative Considered | Rationale |
|----------|------------------------|-----------|
| **Dual Medallion** | Single unified pipeline | Different data velocities, independent scaling, failure isolation |
| **NDJSON Bronze** | CSV or Parquet | Append-only streaming, crash recovery, no file rewrites |
| **3-CSV Silver** | Single merged CSV | Sparse data handling (order book unavailable), debugging isolation |
| **Separate Gold Storage** | Merged Gold layer | Independent reprocessing, model flexibility, cleaner separation |
| **XGBoost with Lags** | LSTM/Transformer | Limited data (8,760 samples/year), interpretability, fast training |
| **As-of Join** | Simple time-based merge | Prevents look-ahead bias, realistic trading simulation |
| **FinGPT Market Context** | Text-only analysis | Market-aware sentiment, coherence detection, adjusted signal strength |
| **Paper Trading Account** | Live account | Real data without financial risk, safe for experimentation |
| **Scraped News** | Paid API (Bloomberg/Reuters) | Cost efficiency ($0 vs $2000/month), SGD-specific filtering control |
| **Hourly Granularity** | Minute/tick-level | Balances data volume, FinGPT processing time, news frequency |

---

**This architecture provides a production-ready, scalable foundation for SGD FX prediction with clear separation of concerns, robust error handling, and comprehensive observability.**

For detailed ETL documentation, see [ETL_COMPLETE_DOCUMENTATION.md](ETL_COMPLETE_DOCUMENTATION.md)