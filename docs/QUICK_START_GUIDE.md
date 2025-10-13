# Quick Start Guide - All 3 Components

This guide will help you set up and test the 3 newly implemented components:
1. ✅ Feast Feature Store Integration
2. ✅ FastAPI Backend
3. ✅ News Simulator

---

## Prerequisites

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific packages
pip install \
    feast>=0.35.0 \
    pyarrow>=12.0.0 \
    redis>=5.0.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    flask>=3.0.0 \
    flask-cors>=4.0.0
```

---

## 1. Feast Feature Store Setup

### Step 1.1: Start Redis

```bash
# Using Docker Compose
docker-compose up -d redis

# Or using Docker directly
docker run -d -p 6379:6379 redis:7-alpine

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### Step 1.2: Generate Parquet Files

The Gold layer scripts now automatically generate Parquet files alongside CSVs:

```bash
# Process S&P 500 data
python run_sp500_pipeline.py --skip-labels

# This will create:
# - data/sp500/gold/training/market_features.parquet
# - data/news/gold/news_signals/sp500_trading_signals.parquet
```

### Step 1.3: Initialize Feast

```bash
# Apply feature definitions to registry
python scripts/feast_materialize.py

# This will:
# 1. Run `feast apply` in feature_repo/
# 2. Materialize features to Redis
# 3. Enable online serving
```

### Step 1.4: Test Feast Online Features

```bash
# Test feature fetching
python scripts/test_feast_online.py

# Expected output:
# ✓ Feast store initialized
# ✓ Successfully fetched market features from Redis
# Features retrieved:
#   ret_1h: [0.0023]
#   ret_4h: [0.0045]
#   ...
```

### Troubleshooting Feast

**Issue**: `feast apply` fails
```bash
# Make sure you're in the feature_repo directory
cd feature_repo
feast apply
cd ..
```

**Issue**: No features materialized
```bash
# Check Parquet files exist
ls -lh data/sp500/gold/training/*.parquet
ls -lh data/news/gold/news_signals/*.parquet

# Check Redis is accessible
redis-cli ping
```

**Issue**: "event_timestamp column not found"
```bash
# Re-run Gold layer scripts (they now add event_timestamp)
python src/build_sp500_gold.py \
    --technical-features data/sp500/silver/technical_features/sp500_technical.csv \
    --microstructure-features data/sp500/silver/microstructure/sp500_microstructure.csv \
    --volatility-features data/sp500/silver/volatility/sp500_volatility.csv \
    --output data/sp500/gold/training/market_features.csv
```

---

## 2. FastAPI Backend Setup

### Step 2.1: Start the API Server

```bash
# Development mode (with auto-reload)
cd src
python -m api.main

# Or using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 2.2: Test Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-10-13T10:00:00",
  "model_loaded": true,
  "feast_available": true,
  "redis_connected": true
}
```

**Get Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instrument": "SPX500_USD"}'

# Expected response:
{
  "instrument": "SPX500_USD",
  "timestamp": "2025-10-13T10:00:00",
  "prediction": "bullish",
  "probability": 0.73,
  "confidence": 0.85,
  "signal_strength": 0.46,
  "features_used": 144,
  "model_version": "gradient_boosting"
}
```

**Get Recent News:**
```bash
curl http://localhost:8000/news/recent?limit=5

# Expected response:
[
  {
    "time": "2025-10-13T09:30:00",
    "headline": "S&P 500 surges on strong earnings",
    "source": "MarketWatch",
    "sentiment": 0.75,
    "impact": "high"
  },
  ...
]
```

**Historical Predictions:**
```bash
curl "http://localhost:8000/predictions/history?instrument=SPX500_USD&hours=24"
```

### Step 2.3: Test WebSocket Streaming

```bash
# Install wscat for WebSocket testing
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:8000/ws/market-stream

# You should see real-time updates every 5 seconds:
{
  "type": "market_update",
  "timestamp": "2025-10-13T10:00:00",
  "data": {
    "instrument": "SPX500_USD",
    "price": 4521.50,
    "prediction": "bullish",
    "probability": 0.73,
    "confidence": 0.85
  }
}
```

### Step 2.4: View API Documentation

Visit http://localhost:8000/docs for interactive Swagger UI

### Troubleshooting FastAPI

**Issue**: Model not loaded
```bash
# Check if model file exists
ls -lh models/*.pkl

# If missing, train the model first:
python src/train_combined_model.py
```

**Issue**: Feast features not available
```bash
# Make sure Feast is initialized and Redis is running
python scripts/feast_materialize.py
docker-compose up -d redis
```

**Issue**: Port 8000 already in use
```bash
# Use a different port
uvicorn src.api.main:app --port 8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

---

## 3. News Simulator Setup

### Step 3.1: Start the Simulator

```bash
# Navigate to news-simulator directory
cd news-simulator

# Start Flask app
python app.py

# The simulator will start on http://localhost:5000
```

### Step 3.2: Access the UI

Open your browser and visit:

```
http://localhost:5000
```

You should see:
- 📰 News Simulator header
- Statistics panel showing available articles
- 3 buttons:
  - 🟢 Stream Positive News
  - 🔴 Stream Negative News
  - ⚪ Stream Neutral News

### Step 3.3: Stream News

1. Click any of the 3 buttons
2. See the "Last Streamed Article" section update
3. Check streamed articles in: `data/news/bronze/simulated/`

### Step 3.4: Test API

```bash
# Get statistics
curl http://localhost:5000/api/stats

# Stream positive news
curl -X POST http://localhost:5000/api/stream/positive

# Stream negative news
curl -X POST http://localhost:5000/api/stream/negative

# Stream neutral news
curl -X POST http://localhost:5000/api/stream/neutral

# Reload articles
curl -X POST http://localhost:5000/api/reload
```

### Troubleshooting News Simulator

**Issue**: No articles available
```bash
# The simulator will generate mock articles automatically
# To load real articles, first run the news scraper:
python src/scrape_sp500_news_free.py

# Or use historical scraper
python src/scrape_historical_sp500_news.py
```

**Issue**: Port 5000 already in use
```bash
# Change port in app.py:
# app.run(port=5001)

# Or kill existing process
lsof -ti:5000 | xargs kill -9
```

**Issue**: CORS errors in browser
- Flask-CORS is already enabled
- Check browser console for specific errors

---

## 4. Test All Components Together

### Terminal 1: Start Redis
```bash
docker-compose up redis
```

### Terminal 2: Start FastAPI
```bash
uvicorn src.api.main:app --reload
```

### Terminal 3: Start News Simulator
```bash
cd news-simulator && python app.py
```

### Terminal 4: Run Tests
```bash
# Test Feast
python scripts/test_feast_online.py

# Test API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"instrument": "SPX500_USD"}'

# Test News Simulator
curl -X POST http://localhost:5000/api/stream/positive
```

---

## 5. Docker Compose (All Services)

Update `docker-compose.yml` to include all services:

```bash
# Start all services
docker-compose up -d redis postgres

# Start API
docker-compose up -d api

# Start News Simulator
docker-compose up -d news-simulator

# View logs
docker-compose logs -f api

# Stop all
docker-compose down
```

---

## 6. Next Steps

### Build Frontend

Create a React frontend that:
- Connects to FastAPI WebSocket (`ws://localhost:8000/ws/market-stream`)
- Displays live predictions
- Shows market charts
- Displays news feed

### Real-time Inference Loop

Create a background service that:
1. Polls OANDA for latest S&P 500 price
2. Pushes features to Feast online store
3. Generates predictions every minute
4. Broadcasts to WebSocket clients

### Model Training

```bash
# Train combined model
python src/train_combined_model.py

# Output: models/gradient_boosting_combined_model.pkl
```

---

## 7. Common Commands Summary

```bash
# Feast
python scripts/feast_materialize.py
python scripts/test_feast_online.py

# FastAPI
uvicorn src.api.main:app --reload
curl http://localhost:8000/health
curl http://localhost:8000/docs

# News Simulator
cd news-simulator && python app.py
open http://localhost:5000

# Docker
docker-compose up -d redis
docker-compose up -d api
docker-compose logs -f api

# Full Pipeline
python run_sp500_pipeline.py --skip-labels
python src/train_combined_model.py
```

---

## 8. Verification Checklist

- [ ] Redis running and accessible
- [ ] Parquet files generated in Gold layer
- [ ] Feast registry initialized (`feast apply`)
- [ ] Features materialized to Redis
- [ ] FastAPI server running on port 8000
- [ ] API health check returns "healthy"
- [ ] WebSocket streaming works
- [ ] News Simulator running on port 5000
- [ ] News streaming works (3 buttons functional)
- [ ] Model loaded successfully

---

## Support

If you encounter issues:

1. Check logs: `docker-compose logs -f [service-name]`
2. Verify prerequisites are installed
3. Ensure all ports are available (6379, 8000, 5000)
4. Check the main documentation in `docs/SYSTEM_ANALYSIS_AND_ROADMAP.md`

---

**Status**: ✅ All 3 Quick Wins Implemented!

- ✅ Feast Feature Store (with Parquet generation & Redis)
- ✅ FastAPI Backend (with WebSocket streaming)
- ✅ News Simulator (with 3-button UI)
