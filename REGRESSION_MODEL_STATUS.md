# Regression Model - Price Prediction

## ✅ Status: ACTIVE

The system now uses a **regression model** for price prediction instead of classification.

### Model Details
- **Type:** XGBRegressor
- **Location:** `models/xgboost_regression_30min_20251026_030337.pkl`
- **Task:** Regression (predicts price change %)
- **Features:** 69 technical + sentiment features
- **Horizon:** 30-minute ahead prediction

### Model Performance
```json
{
  "cv_mean_r2": 0.160,
  "test_r2": -0.073,
  "test_rmse": 0.220,
  "test_mae": 0.108,
  "oot_r2": -0.069,
  "oot_rmse": 0.130
}
```

**Note:** Negative R² means the model performs worse than simply predicting the mean. This is why we blend heavily with news sentiment (90% news, 10% model) for the demo.

---

## 🎯 Prediction Output

### Regression vs Classification

**Regression Model Returns:**
```json
{
  "instrument": "SPX500_USD",
  "task": "regression",
  "prediction": "bullish",  // Direction label
  "predicted_relative_change": 0.0164,  // +1.64% expected change
  "predicted_price": 6612.50,  // Predicted price after 30min
  "signal_strength": 0.0164,
  "confidence": 0.0164
}
```

**Classification Model Would Return:**
```json
{
  "task": "classification",
  "prediction": "bullish",
  "probability": 0.62,  // 62% chance of UP
  "confidence": 0.24,
  "predicted_relative_change": null,
  "predicted_price": null
}
```

---

## 📊 Test Results

### Positive News Test
```
Input: 5 articles, avg sentiment = +0.85
Output:
  - Task: regression
  - Direction: bullish
  - Change: +1.64%
  - Predicted Price: $6,612.50
  ✓ PASS
```

### Negative News Test
```
Input: 5 articles, avg sentiment = -0.86
Output:
  - Task: regression
  - Direction: bearish
  - Change: -0.67%
  - Predicted Price: $6,468.77
  ✓ PASS
```

---

## 🔧 Implementation

### Model Loading Priority
File: `src_clean/api/inference.py` (lines 53-61)

```python
alternative_paths = [
    Path("models/xgboost_regression_30min_20251026_030337.pkl"),  # ← PRIORITIZED
    Path("models/lightgbm_regression_30min_20251026_030405.pkl"),
    Path("models/xgboost_classification_30min_20251101_042201.pkl"),
    ...
]
```

The system automatically loads the first available model from this list.

### Prediction Blending
File: `src_clean/api/inference.py` (lines 197-202)

```python
if task == "regression":
    relative_change = float(self.model.predict(feature_array)[0])

    # Blend with news sentiment
    simulated_sentiment = self._get_simulated_news_sentiment()
    if simulated_sentiment is not None:
        # Weight: 10% model, 90% news (for demo)
        news_relative_change = simulated_sentiment * 0.015
        relative_change = (relative_change * 0.1) + (news_relative_change * 0.9)
```

---

## 🚀 Usage

### Start All Services
```bash
./start_all.sh
```

This will:
1. Load XGBRegressor automatically
2. Start News Simulator on port 5000
3. Start FastAPI on port 8000
4. Start Streamlit on port 8501

### Test Price Prediction

**Generate positive news:**
```bash
curl -X POST http://localhost:5000/api/stream/positive
```

**Get price prediction:**
```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"instrument": "SPX500_USD"}'
```

Expected response:
```json
{
  "task": "regression",
  "prediction": "bullish",
  "predicted_relative_change": 0.0135,
  "predicted_price": 6587.25,
  "signal_strength": 0.0135
}
```

---

## 💡 Why Regression for Price Prediction?

### Advantages
1. **Continuous Output:** Predicts actual price change (e.g., +1.5%, -0.8%)
2. **Price Levels:** Can calculate predicted price: `current_price * (1 + change)`
3. **Magnitude:** Tells you HOW MUCH the price will move, not just direction
4. **Better for Trading:** Can set profit targets, stop losses based on predicted magnitude

### Classification Limitations
- Only predicts UP/DOWN direction
- Doesn't tell magnitude of move
- Can't calculate target price
- 51% up vs 99% up treated the same

---

## 📈 Model Improvements Needed

Current model has poor performance (R² = -0.07). To improve:

1. **More Data:** Train on longer history (currently ~5 years)
2. **Better Features:**
   - Add volume-based indicators
   - Include market breadth metrics
   - Add volatility features (VIX)
3. **Different Horizons:** Try 5min, 15min, 60min predictions
4. **Ensemble:** Combine multiple models
5. **Different Target:**
   - Currently: percentage return
   - Alternative: log returns, z-scored returns
6. **Better Regularization:** Current model may be overfitting

---

## 🎯 Production Deployment

For production use:

1. **Reduce News Weight:** Change from 90% news to 30% news
   ```python
   relative_change = (relative_change * 0.7) + (news_relative_change * 0.3)
   ```

2. **Train Better Model:** Use longer history + better features

3. **Add Confidence Bounds:**
   - Use quantile regression for uncertainty estimates
   - Return prediction intervals

4. **Backtest:** Test on out-of-sample data to verify performance

5. **Real-time Features:**
   - Materialize news to Feast
   - Connect OANDA for live market data

---

## 📝 Summary

✅ Regression model loaded and working
✅ Predicts price changes (not just direction)
✅ Calculates predicted price levels
✅ Blends with news sentiment for demo
✅ Positive news → positive price changes
✅ Negative news → negative price changes

The model is ready for demo purposes!
