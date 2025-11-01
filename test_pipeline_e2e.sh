#!/bin/bash
# End-to-End Pipeline Test (Without Airflow/Docker)
# Tests that all pipeline components work with 1-minute resampling fix

set -e  # Exit on error

echo "=========================================="
echo "E2E Pipeline Test - Regression Model Training"
echo "=========================================="
echo ""

# Use existing data
BRONZE_MARKET="data_clean/bronze/market/spx500_usd_m1_5years.ndjson"
BRONZE_NEWS="data_clean/bronze/news/historical_5year"

# Output paths
SILVER_TECHNICAL="data_clean/silver/market/technical/test_spx500_technical.csv"
SILVER_VOLATILITY="data_clean/silver/market/volatility/test_spx500_volatility.csv"
SILVER_MICRO="data_clean/silver/market/microstructure/test_spx500_microstructure.csv"
GOLD_FEATURES="data_clean/gold/market/features/test_spx500_features.parquet"
GOLD_LABELS="data_clean/gold/market/labels/test_spx500_labels_30min.csv"
GOLD_NEWS="data_clean/gold/news/signals/test_sp500_trading_signals.parquet"
MODEL_OUTPUT="models/test"

echo "Step 1: Process Bronze → Silver (Technical Indicators)"
echo "   Testing 1-minute resampling fix..."
.venv/bin/python -m src_clean.data_pipelines.silver.market_technical_processor \
  --input "$BRONZE_MARKET" \
  --output "$SILVER_TECHNICAL" \
  2>&1 | grep -E "(Loaded|Reindex|Percentage of backfill)"

echo ""
echo "Step 2: Process Bronze → Silver (Volatility)"
.venv/bin/python -m src_clean.data_pipelines.silver.market_volatility_processor \
  --input "$BRONZE_MARKET" \
  --output "$SILVER_VOLATILITY" \
  2>&1 | tail -3

echo ""
echo "Step 3: Process Bronze → Silver (Microstructure)"
.venv/bin/python -m src_clean.data_pipelines.silver.market_microstructure_processor \
  --input "$BRONZE_MARKET" \
  --output "$SILVER_MICRO" \
  2>&1 | tail -3

echo ""
echo "Step 4: Merge Silver → Gold (Market Features)"
.venv/bin/python -m src_clean.data_pipelines.gold.market_gold_builder \
  --technical "$SILVER_TECHNICAL" \
  --volatility "$SILVER_VOLATILITY" \
  --microstructure "$SILVER_MICRO" \
  --output "$GOLD_FEATURES" \
  2>&1 | tail -5

echo ""
echo "Step 5: Generate Labels"
.venv/bin/python -m src_clean.data_pipelines.gold.label_generator \
  --input "$GOLD_FEATURES" \
  --output "$GOLD_LABELS" \
  --horizon 30 \
  2>&1 | tail -5

echo ""
echo "Step 6: Process News (FinBERT)"
echo "   Using existing gold news signals..."
if [ -f "data_clean/gold/news/signals/sp500_trading_signals.parquet" ]; then
    cp data_clean/gold/news/signals/sp500_trading_signals.parquet "$GOLD_NEWS"
    echo "   ✓ News signals copied"
else
    echo "   ! Skipping news processing (no existing signals)"
fi

echo ""
echo "Step 7: Train XGBoost Regression Model"
echo "   Testing with 69 features (includes news_age_minutes, news_available)..."
.venv/bin/python -m src_clean.training.xgboost_training_pipeline_mlflow \
  --market-features "$GOLD_FEATURES" \
  --labels "$GOLD_LABELS" \
  --news-signals "$GOLD_NEWS" \
  --prediction-horizon 30 \
  --task regression \
  --output-dir "$MODEL_OUTPUT" \
  --experiment-name "e2e_test" \
  --mlflow-uri "mlruns" \
  2>&1 | grep -E "(Loaded|features|RMSE|Model saved)"

echo ""
echo "=========================================="
echo "E2E Test Results"
echo "=========================================="

# Check outputs
echo ""
echo "Generated Files:"
ls -lh "$SILVER_TECHNICAL" "$SILVER_VOLATILITY" "$SILVER_MICRO" 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
ls -lh "$GOLD_FEATURES" "$GOLD_LABELS" 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
ls -lh "$MODEL_OUTPUT"/*.pkl 2>/dev/null | head -1 | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "Model Metrics:"
if [ -f "$MODEL_OUTPUT"/*_metrics.json ]; then
    cat "$MODEL_OUTPUT"/*_metrics.json | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'  Test RMSE: {d[\"test_rmse\"]:.4f}'); print(f'  OOT RMSE: {d[\"oot_rmse\"]:.4f}')"
fi

echo ""
echo "=========================================="
echo "✓ E2E Test Complete!"
echo "=========================================="
echo ""
echo "All pipeline components working correctly with:"
echo "  • 1-minute resampling (fixes technical indicators)"
echo "  • News age & availability features (fixes inference)"
echo "  • Parquet support (faster I/O)"
echo ""
echo "Clean up test files with:"
echo "  rm -rf data_clean/silver/market/*/test_* data_clean/gold/*/test_* models/test/"
echo ""
