# System Analysis & Implementation Roadmap

**Date**: October 13, 2025
**Project**: S&P 500 ML Prediction Pipeline
**Status**: Analysis of Current State + Recommendations

---

## Executive Summary

This document provides a comprehensive analysis of the current system architecture, identifies gaps, and proposes an implementation roadmap for a production-ready ML trading system with real-time inference capabilities.

### Current Status Overview

| Component | Status | Completeness | Notes |
|-----------|--------|--------------|-------|
| **Data Collection** | ✅ Working | 90% | Market + News scraping functional |
| **Feature Engineering** | ✅ Working | 85% | 144 features across Bronze→Silver→Gold |
| **Feast Feature Store** | ⚠️ Partial | 40% | Defined but not fully integrated |
| **Model Training** | ✅ Working | 80% | XGBoost training pipeline exists |
| **FastAPI Backend** | ❌ Missing | 0% | Not implemented |
| **Frontend** | ❌ Missing | 0% | Not implemented |
| **Real-time Inference** | ❌ Missing | 0% | Not implemented |
| **News Simulator** | ⚠️ Partial | 30% | Basic simulation exists, no UI |

---

## 1. Feast Feature Store Analysis

### Current Implementation

**Location**: `feature_repo/`

**Files**:
- `feature_store.yaml` - Configuration
- `entities.py` - Entity definitions (instrument)
- `market_features.py` - Market feature view
- `news_signals.py` - News feature view
- `feature_service.py` - Combined service definition

### ✅ What's Working

```yaml
# feature_store.yaml
project: fx_ml
registry: feature_repo/registry.db
provider: local
offline_store:
  type: file
online_store:
  type: redis
  connection_string: redis://localhost:6379
```

**Good**:
- Redis configured for online serving
- File-based offline store for training
- Both market and news feature views defined
- Combined feature service exists
- Docker Compose includes Redis container

### ❌ Critical Gaps

1. **No Parquet Files Generated**
   - Feature views expect `.parquet` files
   - Current pipeline outputs `.csv` files
   - Need conversion step: CSV → Parquet

2. **Feast Not Applied**
   - No `feast apply` in pipeline
   - Registry not initialized
   - Features not materialized to online store

3. **No Integration with Training**
   - `train_combined_model.py` reads CSVs directly
   - Should use `feast.get_historical_features()`

4. **No Online Serving Integration**
   - No code to push features to Redis
   - No code to fetch from Redis for inference

### 🔧 Required Fixes

**Fix 1: Add Parquet Generation**
```python
# In build_market_gold.py and build_news_gold.py
import pyarrow.parquet as pq

# After saving CSV
df.to_parquet(output_path.with_suffix('.parquet'), index=False)
```

**Fix 2: Add Feast Materialization Script**
```python
# scripts/materialize_features.py
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")

# Apply feature definitions
subprocess.run(["feast", "apply"], cwd="feature_repo")

# Materialize to online store
store.materialize_incremental(end_date=datetime.now())
```

**Fix 3: Update Training to Use Feast**
```python
# In train_combined_model.py
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Get historical features
entity_df = pd.DataFrame({
    'instrument': ['SPX500_USD'] * len(timestamps),
    'event_timestamp': timestamps
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        'market_gold_features:ret_1h',
        'news_gold_signals:news_sentiment_score',
        # ... all features
    ]
).to_df()
```

**Fix 4: Add Online Feature Serving**
```python
# For real-time inference
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Fetch online features
features = store.get_online_features(
    features=[
        'market_gold_features:ret_1h',
        'news_gold_signals:news_sentiment_score',
    ],
    entity_rows=[
        {'instrument': 'SPX500_USD'}
    ]
).to_dict()
```

### 📋 Action Items for Feast

- [ ] Install `pyarrow` for Parquet support: `pip install pyarrow`
- [ ] Modify Gold layer scripts to output Parquet
- [ ] Create `scripts/feast_materialize.py`
- [ ] Run `feast apply` to initialize registry
- [ ] Test materialization to Redis
- [ ] Update training script to use Feast offline store
- [ ] Create online feature fetching utility

---

## 2. FastAPI Backend Analysis

### Current Status: ❌ NOT IMPLEMENTED

**Docker Compose shows API service** (line 177-199) but:
- No `src/api.py` file exists
- Dockerfile references it but it's missing
- FastAPI dependencies commented out in requirements.txt

### 🎯 Required Backend API Architecture

```
src/api/
├── __init__.py
├── main.py              # FastAPI app entry point
├── models.py            # Pydantic models
├── inference.py         # Model inference logic
├── feature_fetcher.py   # Feast online feature fetching
└── websocket.py         # WebSocket for streaming data
```

### 📝 API Endpoints Required

#### 1. Health Check
```python
GET /health
Response: {"status": "healthy", "model_loaded": true}
```

#### 2. Model Prediction
```python
POST /predict
Request: {
  "instrument": "SPX500_USD",
  "timestamp": "2025-10-13T10:00:00Z"
}
Response: {
  "prediction": "bullish",
  "probability": 0.73,
  "confidence": 0.85,
  "features_used": 144,
  "model": "xgboost_combined"
}
```

#### 3. Historical Predictions
```python
GET /predictions/history?instrument=SPX500_USD&hours=24
Response: {
  "predictions": [
    {"time": "...", "prediction": "bullish", "probability": 0.73},
    ...
  ]
}
```

#### 4. WebSocket - Live Market Data + Predictions
```python
WS /ws/market-stream
Subscribe: {"instrument": "SPX500_USD"}
Stream: {
  "type": "market_update",
  "data": {
    "time": "2025-10-13T10:00:00Z",
    "price": 4521.50,
    "volume": 1234567,
    "prediction": "bullish",
    "probability": 0.73
  }
}
```

#### 5. News Sentiment Feed
```python
GET /news/recent?limit=10
Response: {
  "news": [
    {
      "time": "...",
      "headline": "...",
      "sentiment": 0.45,
      "source": "MarketWatch",
      "impact": "high"
    }
  ]
}
```

### 🔧 Implementation Template

```python
# src/api/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from .models import PredictionRequest, PredictionResponse
from .inference import ModelInference
from .websocket import MarketStreamManager

app = FastAPI(title="S&P 500 ML Prediction API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model inference
model = ModelInference(
    model_path="models/xgboost_combined_model.pkl",
    feast_repo="feature_repo"
)

# Initialize WebSocket manager
ws_manager = MarketStreamManager()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model.is_loaded}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    return await model.predict(request)

@app.websocket("/ws/market-stream")
async def market_stream(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Stream market data + predictions
            data = await model.get_latest_prediction()
            await websocket.send_json(data)
            await asyncio.sleep(1)  # 1-second intervals
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.get("/news/recent")
async def recent_news(limit: int = 10):
    # Read from data/news/gold/
    return await model.get_recent_news(limit)
```

### 📋 Action Items for Backend API

- [ ] Create `src/api/` directory structure
- [ ] Implement `main.py` with core endpoints
- [ ] Create `inference.py` with XGBoost model loading
- [ ] Integrate Feast online feature fetching
- [ ] Implement WebSocket streaming
- [ ] Add CORS configuration
- [ ] Write API tests
- [ ] Update Docker Compose API service
- [ ] Uncomment FastAPI dependencies in requirements.txt

---

## 3. Frontend Architecture

### Recommended: **Node.js + React + TypeScript**

**Why Node.js?**
- ✅ Best ecosystem for modern web frontends
- ✅ TypeScript for type safety
- ✅ WebSocket support (real-time streaming)
- ✅ Rich charting libraries (Recharts, Victory, TradingView)
- ✅ Fast development with React/Next.js

### 🎨 Frontend Requirements

#### Pages/Views

1. **Dashboard** (`/`)
   - Live S&P 500 price chart
   - Latest prediction (bullish/bearish/neutral)
   - Confidence meter
   - Recent news sentiment

2. **Market Data** (`/market`)
   - Historical price charts (1H, 4H, 1D, 1W)
   - Technical indicators visualization
   - Volume profiles
   - Feature importance display

3. **Predictions** (`/predictions`)
   - Prediction history
   - Model performance metrics
   - Accuracy over time
   - Confusion matrix

4. **News Analysis** (`/news`)
   - Recent news articles
   - Sentiment scores
   - Entity extraction results
   - News impact on predictions

### 🛠️ Tech Stack Recommendation

```json
{
  "frontend": {
    "framework": "React 18 + TypeScript",
    "build": "Vite",
    "routing": "React Router v6",
    "state": "Zustand or Redux Toolkit",
    "websocket": "Socket.IO client or native WebSocket",
    "charts": "Recharts + TradingView Lightweight Charts",
    "styling": "Tailwind CSS",
    "ui": "shadcn/ui or Ant Design"
  },
  "deployment": {
    "dev": "Vite dev server",
    "prod": "Nginx serving static build",
    "docker": "Separate frontend container"
  }
}
```

### 📂 Frontend Structure

```
frontend/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── public/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   ├── client.ts          # Axios/Fetch wrapper
│   │   └── websocket.ts       # WebSocket connection
│   ├── components/
│   │   ├── Dashboard/
│   │   ├── MarketChart/
│   │   ├── PredictionCard/
│   │   └── NewsCard/
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── Market.tsx
│   │   ├── Predictions.tsx
│   │   └── News.tsx
│   ├── hooks/
│   │   ├── useMarketData.ts
│   │   ├── usePredictions.ts
│   │   └── useWebSocket.ts
│   ├── types/
│   │   └── api.ts             # TypeScript types
│   └── utils/
│       └── formatters.ts
└── Dockerfile
```

### 🎯 Key Features to Implement

#### 1. Real-time Market Data Display

```typescript
// src/hooks/useMarketData.ts
import { useEffect, useState } from 'react'

export function useMarketData() {
  const [data, setData] = useState(null)

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/market-stream')

    ws.onmessage = (event) => {
      const marketData = JSON.parse(event.data)
      setData(marketData)
    }

    return () => ws.close()
  }, [])

  return data
}
```

#### 2. Prediction Indicator

- **Visual**: Traffic light system (🟢 Bullish, 🟡 Neutral, 🔴 Bearish)
- **Confidence**: Progress bar (0-100%)
- **Animation**: Smooth transitions when prediction updates

#### 3. Market Chart with Prediction Overlays

```typescript
// Overlay prediction zones on price chart
// - Green background: Predicted bullish zones
// - Red background: Predicted bearish zones
// - Markers: Actual prediction points
```

### 📋 Action Items for Frontend

- [ ] Initialize React + TypeScript project with Vite
- [ ] Set up Tailwind CSS + UI library
- [ ] Create WebSocket hook for market streaming
- [ ] Implement Dashboard page with live price chart
- [ ] Add prediction display with confidence indicator
- [ ] Create news feed component
- [ ] Add Docker configuration for frontend
- [ ] Configure CORS in backend API
- [ ] Deploy frontend to port 3000

---

## 4. News Simulator Application

### Current State

**Exists**: `src/simulate_news_feed.py`
- Reads historical news from files
- Can simulate streaming
- No UI, command-line only

### 🎯 Required: Standalone Simulator with UI

```
news-simulator/
├── backend/
│   ├── app.py              # Flask/FastAPI lightweight server
│   ├── news_reader.py      # Read scraped news files
│   └── streamer.py         # Stream news to Bronze pipeline
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── Dockerfile
```

### 🖥️ UI Requirements

**Simple Single-Page Interface**:

```
┌─────────────────────────────────────────┐
│         📰 News Simulator               │
├─────────────────────────────────────────┤
│                                          │
│  Available News Articles: 1,247         │
│  Current Position: 45                   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │  🟢 Stream Positive News         │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │  🔴 Stream Negative News         │   │
│  └──────────────────────────────────┘   │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │  ⚪ Stream Neutral News          │   │
│  └──────────────────────────────────┘   │
│                                          │
│  Last Streamed:                         │
│  [2025-10-13 10:45:23] Fed signals...  │
│  Sentiment: 0.72 (Positive)             │
│                                          │
│  Status: ✅ Connected to Bronze API    │
└─────────────────────────────────────────┘
```

### 🔧 Implementation

```python
# news-simulator/backend/app.py
from flask import Flask, render_template, jsonify, request
import pandas as pd
from pathlib import Path
import requests

app = Flask(__name__)

# Load scraped news
NEWS_DIR = Path("../data/news/bronze/raw_articles/")
news_data = []

def load_news():
    # Load all scraped news files
    for file in NEWS_DIR.glob("*.json"):
        with open(file) as f:
            news_data.append(json.load(f))

def filter_by_sentiment(sentiment_type):
    """Filter news by sentiment: positive, negative, neutral"""
    if sentiment_type == "positive":
        return [n for n in news_data if n.get('sentiment_score', 0) > 0.3]
    elif sentiment_type == "negative":
        return [n for n in news_data if n.get('sentiment_score', 0) < -0.3]
    else:
        return [n for n in news_data if abs(n.get('sentiment_score', 0)) <= 0.3]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream/<sentiment_type>')
def stream_news(sentiment_type):
    """Stream a news article of specified sentiment to Bronze pipeline"""
    filtered = filter_by_sentiment(sentiment_type)

    if not filtered:
        return jsonify({"error": "No news available"}), 404

    # Pick random article
    article = random.choice(filtered)

    # Send to Bronze pipeline API
    response = requests.post(
        "http://localhost:8000/ingest/news",
        json=article
    )

    return jsonify({
        "status": "streamed",
        "article": article,
        "response": response.json()
    })

if __name__ == '__main__':
    load_news()
    app.run(port=5000)
```

### 📋 Action Items for News Simulator

- [ ] Create `news-simulator/` directory
- [ ] Implement Flask/FastAPI backend
- [ ] Create simple HTML/CSS/JS frontend
- [ ] Add sentiment filtering logic
- [ ] Implement Bronze pipeline ingestion endpoint
- [ ] Add WebSocket for status updates
- [ ] Dockerize simulator
- [ ] Document usage

---

## 5. Pipeline Architecture Assessment

### Current Architecture: ✅ SOLID

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                         │
├─────────────────────────────────────────────────────────┤
│  OANDA API (Market)  │  RSS/News APIs (News)            │
└────────┬─────────────┴──────────────┬──────────────────┘
         │                            │
         ▼                            ▼
    ┌─────────┐                  ┌─────────┐
    │ BRONZE  │                  │ BRONZE  │
    │ Market  │                  │  News   │
    │ (NDJSON)│                  │ (JSON)  │
    └────┬────┘                  └────┬────┘
         │                            │
         ▼                            ▼
    ┌─────────┐                  ┌─────────┐
    │ SILVER  │                  │ SILVER  │
    │ Technical│                 │Sentiment│
    │ Features │                 │ Entity  │
    │ (37 feat)│                 │ (95 feat)│
    └────┬────┘                  └────┬────┘
         │                            │
         ▼                            ▼
    ┌─────────┐                  ┌─────────┐
    │  GOLD   │                  │  GOLD   │
    │ Training│                  │ Signals │
    │ Ready   │                  │ (11 f.) │
    └────┬────┘                  └────┬────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
              ┌──────────────┐
              │   COMBINED   │
              │   TRAINING   │
              │  144 Features│
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │   XGBOOST    │
              │    MODEL     │
              └──────────────┘
```

### ✅ Strengths

1. **Medallion Architecture**: Clean Bronze→Silver→Gold separation
2. **Dual Pipelines**: Market and News processed independently
3. **Feature-Rich**: 144 total features
4. **Modular Code**: Each stage is a separate script
5. **Docker Support**: All services containerized
6. **Orchestration**: `orchestrate_pipelines.py` coordinates execution

### ⚠️ Areas for Improvement

#### 1. Missing Real-time Streaming Layer

**Current**: Batch processing only
**Needed**: Streaming pipeline for live inference

**Suggestion**: Add streaming layer

```
OANDA Stream → Kafka → Stream Processor → Feast → Inference
```

**Alternative** (simpler): Polling-based

```python
# src/streaming/live_inference.py
while True:
    # Poll OANDA for latest candle
    latest_candle = fetch_latest_candle()

    # Push to Feast online store
    feast_store.push_online_features(...)

    # Get online features + predict
    features = feast_store.get_online_features(...)
    prediction = model.predict(features)

    # Push to API/WebSocket clients
    await broadcast_prediction(prediction)

    time.sleep(60)  # Every 1 minute
```

#### 2. No Label Generation for Training

**Issue**: `train_combined_model.py` expects a 'y' column but no script generates it

**Solution**: Implement `src/build_labels.py` (exists but incomplete)

```python
# Label strategy: Forward-looking returns
# y = 1 if price increases >0.5% in next 1 hour
# y = 0 otherwise

def generate_labels(df, horizon='1H', threshold=0.005):
    df['future_return'] = df['close'].pct_change(periods=60).shift(-60)
    df['y'] = (df['future_return'] > threshold).astype(int)
    return df
```

#### 3. Feast Integration Incomplete

**See Section 1 above for full details**

#### 4. No Model Versioning or Registry

**Suggestion**: Add MLflow or simple versioning

```python
# models/
# ├── v1.0.0_xgboost_combined_2025-10-13.pkl
# ├── v1.0.1_xgboost_combined_2025-10-14.pkl
# └── registry.json

{
  "models": [
    {
      "version": "1.0.1",
      "path": "v1.0.1_xgboost_combined_2025-10-14.pkl",
      "trained_at": "2025-10-14T10:00:00Z",
      "metrics": {
        "auc": 0.73,
        "accuracy": 0.68
      },
      "active": true
    }
  ]
}
```

#### 5. No Monitoring or Logging

**Suggestion**: Add structured logging

```python
# Use Python's logging + file rotation
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/pipeline.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
logging.basicConfig(handlers=[handler])
```

### 📋 Pipeline Improvement Action Items

- [ ] Implement real-time streaming inference loop
- [ ] Complete `build_labels.py` for training targets
- [ ] Add model versioning and registry
- [ ] Implement structured logging
- [ ] Add data quality checks (Great Expectations)
- [ ] Create monitoring dashboard (Grafana)
- [ ] Add alerting for pipeline failures
- [ ] Implement CI/CD for model deployment

---

## 6. Complete Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)

**Priority: Fix Feast Integration**

- [ ] Add Parquet output to Gold layer scripts
- [ ] Install `pyarrow`: `pip install pyarrow`
- [ ] Run `feast apply` to initialize registry
- [ ] Test offline feature fetching
- [ ] Test online feature materialization to Redis

**Priority: Implement FastAPI Backend**

- [ ] Create `src/api/` structure
- [ ] Implement core endpoints (health, predict)
- [ ] Integrate Feast online features
- [ ] Test with Postman/curl
- [ ] Update Docker Compose

### Phase 2: Frontend + Real-time (Week 2)

**Priority: Build Frontend**

- [ ] Initialize React + TypeScript project
- [ ] Set up WebSocket connection
- [ ] Implement Dashboard page
- [ ] Add market data charts
- [ ] Add prediction display

**Priority: Real-time Inference**

- [ ] Create `src/streaming/live_inference.py`
- [ ] Implement polling-based market data fetch
- [ ] Push features to Feast online store
- [ ] Generate predictions every 1 minute
- [ ] Broadcast via WebSocket

### Phase 3: News Simulator (Week 3)

**Priority: Build News Simulator**

- [ ] Create Flask/FastAPI backend
- [ ] Implement 3-button UI
- [ ] Add sentiment filtering
- [ ] Create Bronze pipeline ingestion endpoint
- [ ] Test end-to-end news streaming

### Phase 4: Production Readiness (Week 4)

**Priority: Monitoring & Operations**

- [ ] Add structured logging
- [ ] Implement model versioning
- [ ] Create monitoring dashboard
- [ ] Add alerting
- [ ] Write deployment documentation
- [ ] Load testing
- [ ] Security hardening

---

## 7. Quick Wins (Do First)

### 1. Fix Feast (2 hours)

```bash
# Add to requirements.txt
echo "pyarrow>=12.0.0" >> requirements.txt
pip install pyarrow

# Modify Gold layer scripts
# Add: df.to_parquet(output_path.with_suffix('.parquet'))

# Initialize Feast
cd feature_repo
feast apply
```

### 2. Create Basic FastAPI (3 hours)

```bash
# Create API directory
mkdir -p src/api

# Create main.py with health + predict endpoints
# Test with curl

# Update Docker Compose to use new API
```

### 3. Build News Simulator UI (4 hours)

```bash
# Create simple Flask app with 3 buttons
# Test streaming to Bronze pipeline
```

---

## 8. Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Collection** | Python + OANDA API | Market data |
| | Python + RSS/News APIs | News data |
| **Processing** | Python + Pandas | ETL pipelines |
| **Feature Store** | Feast + Redis | Online/Offline features |
| **ML Training** | XGBoost + scikit-learn | Model training |
| **Backend API** | FastAPI + WebSocket | Real-time serving |
| **Frontend** | React + TypeScript | Web UI |
| **Database** | PostgreSQL | Metadata |
| **Containerization** | Docker + Docker Compose | Deployment |
| **Orchestration** | Python scripts | Pipeline coordination |

---

## 9. Questions Answered

### Q: Does Feast have all offline features for XGBoost training?

**Answer**: ⚠️ Partially
- Feature views are defined ✅
- Parquet files not generated ❌
- Training script not using Feast ❌
- **Action**: Follow Section 1 to fix

### Q: Does Feast have online store for inference?

**Answer**: ⚠️ Configured but not used
- Redis configured in `feature_store.yaml` ✅
- Docker Compose includes Redis ✅
- No materialization implemented ❌
- No online fetching in code ❌
- **Action**: Follow Section 1 to implement

### Q: Is the backend FastAPI done?

**Answer**: ❌ No, not implemented
- Docker Compose references it but file doesn't exist
- **Action**: Follow Section 2 to build

### Q: Does it have interface for frontend?

**Answer**: ❌ No frontend exists
- **Action**: Follow Section 3 to build React frontend

### Q: How to indicate raw market data vs prediction?

**Recommendation**: Use visual indicators
- **Chart**: Different line colors (blue = actual, green/red = predicted)
- **Background**: Shaded prediction zones on chart
- **Cards**: Separate "Current Price" vs "Prediction" cards
- **Confidence**: Show confidence % next to prediction

### Q: Can frontend be done in Node.js?

**Answer**: ✅ Yes, strongly recommended
- React + TypeScript is the best choice
- Node.js for development server only
- Production: Build static files, serve with Nginx
- **Action**: Follow Section 3

### Q: News simulator with 3-button interface?

**Answer**: ⚠️ Partial implementation exists
- `src/simulate_news_feed.py` has simulation logic
- No UI currently
- **Action**: Follow Section 4 to build UI

### Q: Backend supplies live S&P 500 + predictions to frontend?

**Answer**: ❌ Not implemented
- Need WebSocket endpoint in FastAPI
- Need real-time inference loop
- **Action**: Follow Section 2 + streaming section

### Q: Pipeline correctness and suggestions?

**Answer**: ✅ Pipeline architecture is solid
- Medallion Bronze→Silver→Gold is excellent
- 144 features is comprehensive
- Dual pipelines (market + news) is good design
- **Issues**: See Section 5 for improvements
- **Key gaps**: Real-time streaming, Feast integration, label generation

---

## 10. Next Steps

**Immediate Actions (Today)**:

1. Fix Feast integration (2 hours)
2. Add Parquet generation (30 minutes)
3. Create basic FastAPI (3 hours)

**This Week**:

1. Build React frontend
2. Implement WebSocket streaming
3. Create news simulator UI

**Next Week**:

1. Add monitoring
2. Model versioning
3. Production deployment

---

## Appendix: Example Commands

### Initialize Feast
```bash
cd feature_repo
feast apply
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### Run Full Pipeline
```bash
# Download data
python src/download_sp500_historical.py --years 5

# Process market
python run_sp500_pipeline.py --skip-labels

# Process news
python src/build_news_features.py
python src/build_news_gold.py

# Train model
python src/train_combined_model.py

# Start API
uvicorn src.api.main:app --reload
```

### Start All Services
```bash
docker-compose up redis postgres
docker-compose up api
docker-compose up frontend
docker-compose up news-simulator
```

---

**End of Analysis**
