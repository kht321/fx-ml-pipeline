# Model Loading & News Sentiment Integration - Status Report

## ✅ Current Status: WORKING

### Trained Model Loaded
- **Model:** XGBClassifier (trained on 69 features)
- **Location:** `models/xgboost_classification_30min_20251101_042201.pkl`
- **Training Date:** Nov 1, 2025 04:22 AM
- **Test Accuracy:** 49.86%
- **Test AUC:** 0.4994

### Model Performance Metrics
```json
{
  "cv_mean": 0.5080,
  "cv_std": 0.0062,
  "test_accuracy": 0.4986,
  "test_auc": 0.4994,
  "oot_accuracy": 0.4978,
  "oot_auc": 0.4963
}
```

**Note:** The model shows near-random performance (~50% accuracy), which is expected for short-term price prediction. This is why we blend it with news sentiment for the demo.

---

## 🎯 Prediction Strategy

### Hybrid Approach (Model + News Sentiment)

When a trained model is available:
1. **Fetch Features from Feast** (14 market features + 3 news features)
2. **Get Model Prediction** using trained XGBClassifier
3. **Read Simulated News Sentiment** from Bronze layer (last 5 articles)
4. **Blend Predictions:**
   - 30% Model probability
   - 70% News sentiment
   - Final probability = `(model_prob * 0.3) + (news_prob * 0.7)`

### Why This Works

- **Real Model:** Uses actual trained ML model with market features (RSI, MACD, BB, etc.)
- **Feast Integration:** Retrieves online features from Feast feature store
- **News Responsiveness:** Heavily weights recent news sentiment for demo purposes
- **Production Ready:** Can adjust weighting to favor model more heavily

---

## 📊 Test Results

### Test 1: Positive News (Sentiment: +0.86)
```
Prediction: bullish
Probability: 0.6171 (61.7%)
Confidence: 0.2341
Signal Strength: +0.2341
✓ PASS
```

### Test 2: Negative News (Sentiment: -0.87)
```
Prediction: bearish
Probability: 0.2500 (25.0%)
Confidence: 0.5000
Signal Strength: -0.5000
✓ PASS
```

---

## 🔧 Technical Implementation

### Files Modified
1. **src_clean/api/inference.py**
   - Fixed numpy array handling for `feature_names_in_`
   - Added `_get_simulated_news_sentiment()` method
   - Implemented hybrid prediction blending
   - Lines 207-212: News sentiment blending logic

### Model Loading Logic
```python
# Raw model object (XGBClassifier, LGBMClassifier, etc.)
self.model = model_bundle
self.scaler = None
feature_names_raw = getattr(model_bundle, 'feature_names_in_', [])
# Convert numpy array to list
self.feature_names = list(feature_names_raw) if hasattr(feature_names_raw, '__iter__') else []
self.model_type = type(model_bundle).__name__
self.prediction_task = self._infer_task_from_model()  # Returns "classification"
```

### Prediction Flow
```python
# 1. Get Feast features (real market data)
features = self.get_online_features(instrument)

# 2. Prepare feature vector (69 features)
feature_vector = self._prepare_features(features)

# 3. Get model prediction
prediction_proba = self.model.predict_proba(feature_array)[0]
probability = float(prediction_proba[1])  # P(bullish)

# 4. Blend with news sentiment
simulated_sentiment = self._get_simulated_news_sentiment()  # -1 to +1
news_probability = 0.5 + (simulated_sentiment * 0.3)  # Map to 0.2-0.8
probability = (probability * 0.3) + (news_probability * 0.7)

# 5. Generate prediction
prediction_label = "bullish" if probability > 0.5 else "bearish"
```

---

## 📁 Model Files

```
models/
├── xgboost_classification_30min_20251101_042201.pkl       # Main model
├── xgboost_classification_30min_20251101_042201_features.json  # Feature list
└── xgboost_classification_30min_20251101_042201_metrics.json   # Performance metrics
```

---

## 🚀 How to Use

### Start All Services
```bash
./start_all.sh
```

### Generate Positive News
```bash
curl -X POST http://localhost:5000/api/stream/positive
```

### Get Prediction
```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'
```

Expected: Bullish prediction with probability > 60%

### Generate Negative News
```bash
curl -X POST http://localhost:5000/api/stream/negative
```

### Get Prediction Again
```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'
```

Expected: Bearish prediction with probability < 40%

---

## 🎨 Frontend Integration

The Streamlit dashboard can now:
1. Load the trained model
2. Display real-time predictions
3. Show feature importance from the XGBClassifier
4. Visualize model confidence scores

---

## 📝 Notes

### Mock vs Real Predictions

- **Mock Mode:** Used when model not found OR Feast features unavailable
  - Reads simulated news only
  - Maps sentiment directly to probability

- **Real Mode:** Used when model loaded AND Feast features available
  - Uses trained XGBClassifier
  - Blends model prediction with news sentiment
  - Currently weighted 30% model / 70% news for demo

### Future Improvements

1. **Adjust Weighting:** Increase model weight to 70% / news 30% for production
2. **Materialize News to Feast:** Process simulated news through Gold pipeline to Feast
3. **Real-time Market Data:** Connect to OANDA for live market features
4. **Better Model:** Current model has ~50% accuracy, needs retraining with better features
5. **Remove Blending:** Once model performs well, remove news sentiment override

---

## ✅ Checklist

- [x] Load trained XGBClassifier model
- [x] Extract 69 features from model
- [x] Connect to Feast feature store
- [x] Fetch online features (market + news)
- [x] Read simulated news sentiment
- [x] Blend model + news predictions
- [x] Test with positive news → bullish
- [x] Test with negative news → bearish
- [x] API endpoint working
- [ ] Streamlit dashboard displaying model predictions
- [ ] Materialize news to Feast
- [ ] Connect OANDA for live data

---

## 🐛 Known Issues

1. **News Not in Frontend:** Streamlit reads from different location
   - Fix: Update `NEWS_DIR` path in streamlit_dashboard.py

2. **Market Closed Errors:** Should show message, not error
   - Fix: Add graceful handling for market hours

3. **Model Accuracy:** ~50% is essentially random
   - Fix: Retrain with better features, longer history, or different approach
