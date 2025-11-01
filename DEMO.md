# S&P 500 ML Pipeline - Complete Demo Guide

## 🎯 Overview

This guide walks you through a complete demonstration of the S&P 500 ML prediction pipeline, from setup to live predictions with news sentiment integration.

**What you'll see:**
- Trained XGBoost regression model predicting price changes
- Real-time news sentiment influencing predictions
- Interactive Streamlit dashboard
- FastAPI REST endpoints
- Complete MLOps pipeline

---

## ⚡ Quick Start (2 Minutes)

### Prerequisites
```bash
# Required
- Python 3.11+
- 8GB RAM minimum
- macOS, Linux, or WSL2

# Check Python version
python3.11 --version
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/kht321/fx-ml-pipeline.git
cd fx-ml-pipeline

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Start the Demo

```bash
# Start all services (News Simulator + API + Dashboard)
./start_all.sh
```

This will start:
1. **News Simulator** (port 5000) - Generate mock news articles
2. **FastAPI** (port 8000) - ML prediction service
3. **Streamlit Dashboard** (port 8501) - Interactive frontend

Wait ~10 seconds for all services to initialize.

### Access the Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit Dashboard** | http://localhost:8501 | Main interface |
| **News Simulator** | http://localhost:5000 | Generate test news |
| **FastAPI Docs** | http://localhost:8000/docs | API documentation |
| **Health Check** | http://localhost:8000/health | System status |

---

## 🧪 Testing the Pipeline

### Test 1: Positive News → Bullish Prediction

**Step 1:** Open News Simulator
```bash
open http://localhost:5000
```

**Step 2:** Generate Positive News
- Click **"Positive News"** button 3-5 times
- Each click creates a bullish article (sentiment: +0.5 to +0.9)

**Step 3:** Check Prediction
```bash
# Via API
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'
```

**Expected Response:**
```json
{
  "task": "regression",
  "prediction": "bullish",
  "predicted_relative_change": 0.0135,  // +1.35% expected change
  "predicted_price": 6587.25,
  "signal_strength": 0.0135,
  "confidence": 0.0135
}
```

**Step 4:** View in Dashboard
```bash
open http://localhost:8501
```
- Should show bullish prediction
- Predicted price increase
- Positive sentiment indicator

---

### Test 2: Negative News → Bearish Prediction

**Step 1:** Generate Negative News
- In News Simulator, click **"Negative News"** button 3-5 times
- Each click creates a bearish article (sentiment: -0.9 to -0.5)

**Step 2:** Check Prediction
```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'
```

**Expected Response:**
```json
{
  "task": "regression",
  "prediction": "bearish",
  "predicted_relative_change": -0.0067,  // -0.67% expected decline
  "predicted_price": 6468.77,
  "signal_strength": -0.0067
}
```

**Step 3:** Verify in Dashboard
- Prediction changed to bearish
- Price decrease shown
- Negative sentiment indicator

---

## 📊 Understanding the Output

### Regression Model Output

The system uses **XGBRegressor** for price prediction:

```json
{
  "task": "regression",                    // Model type
  "prediction": "bullish",                 // Direction label
  "predicted_relative_change": 0.0135,     // +1.35% change
  "predicted_price": 6587.25,              // Target price
  "signal_strength": 0.0135,               // Confidence
  "model_version": "XGBRegressor",         // Model name
  "features_used": 69                      // Feature count
}
```

### Key Metrics Explained

| Metric | Range | Meaning |
|--------|-------|---------|
| `predicted_relative_change` | -1.0 to +1.0 | Expected % price change (e.g., 0.015 = +1.5%) |
| `predicted_price` | Price level | Forecasted price after 30 minutes |
| `signal_strength` | -1.0 to +1.0 | Strength of signal (same as change) |
| `confidence` | 0.0 to 1.0 | Model confidence (abs value of change) |

---

## 🎨 Streamlit Dashboard Tour

### Main Features

**1. Live Price Chart**
- Real-time S&P 500 price (from Feast)
- Historical candlestick data
- Predicted price overlay

**2. Prediction Panel**
- Current prediction (bullish/bearish)
- Confidence score
- Expected price change %
- Target price level

**3. News Sentiment**
- Recent news articles
- Sentiment scores
- Impact on predictions

**4. Model Metrics**
- Feature importance
- Model performance stats
- Prediction history

### How to Use

1. **Generate News:**
   - Use News Simulator web UI
   - Or use API: `curl -X POST http://localhost:5000/api/stream/positive`

2. **View Predictions:**
   - Automatically updates in Streamlit
   - Or call: `curl -X POST http://localhost:8000/predict`

3. **Test Different Scenarios:**
   - All positive news → strong bullish
   - All negative news → strong bearish
   - Mixed news → neutral/weak signal

---

## 🔧 Advanced Usage

### API Examples

**Get Prediction:**
```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'
```

**Check Health:**
```bash
curl http://localhost:8000/health
```

**Get Recent News:**
```bash
curl http://localhost:8000/news/recent
```

### News Simulator API

**Stream Positive News:**
```bash
curl -X POST http://localhost:5000/api/stream/positive
```

**Stream Negative News:**
```bash
curl -X POST http://localhost:5000/api/stream/negative
```

**Stream Neutral News:**
```bash
curl -X POST http://localhost:5000/api/stream/neutral
```

**Check Stats:**
```bash
curl http://localhost:5000/api/stats
```

---

## 🛠️ How It Works

### Architecture

```
┌─────────────────┐
│ News Simulator  │ → Generates mock news with sentiment scores
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bronze Layer    │ → Stores news JSON files
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Inference       │ → Reads latest 5 news articles
│ Engine          │   Averages sentiment
└────────┬────────┘   Blends with model prediction
         │
         ▼
┌─────────────────┐
│ XGBRegressor    │ → Predicts price change %
│ (69 features)   │   10% model + 90% news sentiment
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FastAPI         │ → Returns prediction JSON
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Streamlit UI    │ → Displays results
└─────────────────┘
```

### Model Details

**Model:** XGBoost Regression
**Features:** 69 technical indicators from Feast
- RSI, MACD, Bollinger Bands
- SMA, EMA, ATR, ADX
- Volatility metrics
- News sentiment (from Feast)

**Prediction Blending:**
```python
# 10% from trained model, 90% from news sentiment
final_change = (model_prediction * 0.1) + (news_sentiment * 0.015 * 0.9)
```

**Why Heavy News Weighting?**
- Model has poor standalone performance (R² = -0.07)
- Makes demo more responsive to news
- Can be adjusted for production (e.g., 70% model, 30% news)

---

## 📝 Logs & Monitoring

### View Logs

```bash
# All services
tail -f logs/news_simulator.log
tail -f logs/fastapi.log
tail -f logs/streamlit.log

# Or check service status
lsof -ti:5000  # News Simulator
lsof -ti:8000  # FastAPI
lsof -ti:8501  # Streamlit
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "model_loaded": true,
  "feast_available": true,
  "timestamp": "2025-11-01T12:00:00"
}
```

---

## 🐛 Troubleshooting

### Services Won't Start

```bash
# Check ports
lsof -ti:5000 | xargs kill -9  # Kill News Simulator
lsof -ti:8000 | xargs kill -9  # Kill FastAPI
lsof -ti:8501 | xargs kill -9  # Kill Streamlit

# Restart
./start_all.sh
```

### No Predictions Changing

**Problem:** Predictions don't respond to new news

**Solution:**
```bash
# Check if news files are being created
ls -la data_clean/bronze/news/simulated/

# Verify news has sentiment scores
cat data_clean/bronze/news/simulated/simulated_*.json | python3 -m json.tool

# Restart services
./stop_all.sh && ./start_all.sh
```

### Dashboard Not Loading

**Problem:** Streamlit shows errors

**Solution:**
```bash
# Check logs
tail -f logs/streamlit.log

# Restart Streamlit only
pkill -f "streamlit run"
./start_streamlit.sh
```

### Model Not Loading

**Problem:** Always shows "mock" predictions

**Solution:**
```bash
# Check if model file exists
ls -la models/xgboost_regression_*.pkl

# Should auto-load from alternative paths in inference.py
# If missing, copy from data_clean_5year/models/
```

---

## 🎓 Next Steps

### For Learning
1. Review code in `src_clean/api/inference.py`
2. Understand blending logic (lines 197-202)
3. Explore Feast feature store integration
4. Study regression vs classification differences

### For Production
1. Adjust model/news weighting (increase model to 70%)
2. Train better model with more data
3. Add real-time market data (OANDA API)
4. Materialize news through Gold layer to Feast
5. Add confidence intervals (quantile regression)

### Documentation
- **Technical Details:** [REGRESSION_MODEL_STATUS.md](REGRESSION_MODEL_STATUS.md)
- **Startup Guide:** [QUICKSTART.md](QUICKSTART.md)
- **Model Status:** [MODEL_LOADING_STATUS.md](MODEL_LOADING_STATUS.md)

---

## 🛑 Stopping the Demo

```bash
# Stop all services
./stop_all.sh

# Or individually
pkill -f "news-simulator/app.py"
pkill -f "uvicorn src_clean.api.main"
pkill -f "streamlit run"
```

---

## 📚 Additional Resources

**Repository:** https://github.com/kht321/fx-ml-pipeline

**Documentation:**
- README.md - Project overview
- DEMO.md - This file
- QUICKSTART.md - Quick reference
- REGRESSION_MODEL_STATUS.md - Model details

**Support:**
- Check logs in `logs/` directory
- Review GitHub issues
- See troubleshooting section above

---

**Last Updated:** November 1, 2025
**Status:** ✅ Fully Operational
**Demo Time:** ~5 minutes
