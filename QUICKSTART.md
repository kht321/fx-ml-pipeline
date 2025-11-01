# Quick Start Guide - S&P 500 ML Pipeline

## 🚀 Start All Services (Recommended)

### One Command to Start Everything:
```bash
./start_all.sh
```

This starts:
1. **News Simulator** (port 5000) - Generate mock news articles
2. **FastAPI** (port 8000) - ML prediction service
3. **Streamlit** (port 8501) - Interactive dashboard

### Stop All Services:
```bash
./stop_all.sh
```

---

## 📊 Access the Services

### 1. Streamlit Dashboard
**URL:** http://localhost:8501

**Features:**
- Real-time S&P 500 price charts
- ML predictions with confidence scores
- News sentiment visualization
- Feature importance
- Model metrics

### 2. News Simulator
**URL:** http://localhost:5000

**Features:**
- Web interface with 3 buttons:
  - 🟢 Stream Positive News (bullish)
  - 🔴 Stream Negative News (bearish)
  - ⚪ Stream Neutral News

**API Usage:**
```bash
# Generate positive news
curl -X POST http://localhost:5000/api/stream/positive

# Generate negative news
curl -X POST http://localhost:5000/api/stream/negative

# Check stats
curl http://localhost:5000/api/stats
```

### 3. FastAPI Prediction Service
**URL:** http://localhost:8000/docs (Swagger UI)

**API Usage:**
```bash
# Get prediction
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'

# Health check
curl http://localhost:8000/health
```

---

## 🧪 Test the Complete Pipeline

### Step 1: Start all services
```bash
./start_all.sh
```

### Step 2: Generate positive news
Visit http://localhost:5000 and click "Positive News" button 3-5 times

OR use the API:
```bash
for i in {1..5}; do
  curl -X POST http://localhost:5000/api/stream/positive
  sleep 1
done
```

### Step 3: Get prediction (should be bullish)
```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'
```

Expected result:
```json
{
  "prediction": "bullish",
  "probability": 0.75,
  "confidence": 0.50,
  "signal_strength": 0.50
}
```

### Step 4: Generate negative news
Click "Negative News" button 3-5 times

### Step 5: Get prediction again (should be bearish)
```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'
```

Expected result:
```json
{
  "prediction": "bearish",
  "probability": 0.25,
  "confidence": 0.50,
  "signal_strength": -0.50
}
```

---

## 📁 Individual Service Scripts

If you want to start services individually:

### Start News Simulator Only
```bash
./start_news_simulator.sh
```

### Start Streamlit Only
```bash
./start_streamlit.sh
```

### Start FastAPI Only
```bash
.venv/bin/uvicorn src_clean.api.main:app --host 0.0.0.0 --port 8000
```

---

## 🔍 View Logs

```bash
# News Simulator logs
tail -f logs/news_simulator.log

# FastAPI logs
tail -f logs/fastapi.log

# Streamlit logs
tail -f logs/streamlit.log
```

---

## 🐛 Troubleshooting

### Port already in use
```bash
# Check what's using a port
lsof -ti:5000  # News Simulator
lsof -ti:8000  # FastAPI
lsof -ti:8501  # Streamlit

# Kill process on a specific port
lsof -ti:5000 | xargs kill -9
```

### Services won't start
```bash
# Stop everything and try again
./stop_all.sh
sleep 3
./start_all.sh
```

### Check if services are running
```bash
# Check News Simulator
curl http://localhost:5000/api/stats

# Check FastAPI
curl http://localhost:8000/health

# Check Streamlit (should return HTML)
curl http://localhost:8501
```

### View simulated news articles
```bash
# List all articles
ls -lh data_clean/bronze/news/simulated/

# View latest article
ls -t data_clean/bronze/news/simulated/*.json | head -1 | xargs cat | python3 -m json.tool
```

---

## 📚 Additional Resources

- **API Documentation:** http://localhost:8000/docs
- **Inference Engine:** `src_clean/api/inference.py`
- **Dashboard Code:** `src_clean/ui/streamlit_dashboard.py`
- **News Simulator:** `docker/tools/news-simulator/app.py`

---

## ✨ Features

### News → Prediction Flow
1. News Simulator generates articles with sentiment scores
2. Articles saved to `data_clean/bronze/news/simulated/`
3. Inference engine reads 5 most recent articles
4. Average sentiment mapped to prediction probability
5. Probability > 0.5 → bullish, < 0.5 → bearish

### Sentiment Mapping
- Positive news (sentiment: +0.5 to +0.9) → ~70-80% bullish
- Negative news (sentiment: -0.9 to -0.5) → ~20-30% bullish (bearish)
- Neutral news (sentiment: -0.2 to +0.2) → ~45-55% (low confidence)

---

## 🎯 Next Steps

1. **Train a Real Model:** Run training pipeline with historical data
2. **Connect to Live Data:** Configure OANDA API for real market data
3. **Enable Feast:** Materialize features to online store
4. **Deploy:** Use Docker Compose for production deployment
