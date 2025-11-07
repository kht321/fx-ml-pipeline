#!/bin/bash
# Process 5-Year Historical Data Through Medallion Architecture
# Repository: fx-ml-pipeline

set -e  # Exit on error

echo "================================================================================"
echo "5-YEAR DATA PROCESSING PIPELINE"
echo "================================================================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Define paths
BRONZE_MARKET="data_clean/bronze/market/spx500_usd_m1_5years.ndjson"
BRONZE_NEWS="data_clean/bronze/news/historical_5year"
OUTPUT_DIR="data_clean_5year"
INSTRUMENT="spx500"

echo "Input:"
echo "  Market: $BRONZE_MARKET"
echo "  News: $BRONZE_NEWS"
echo "  Output: $OUTPUT_DIR"
echo ""

# ============================================================================
# STAGE 1: BRONZE → SILVER (MARKET)
# ============================================================================
echo "================================================================================"
echo "STAGE 1: BRONZE → SILVER (MARKET)"
echo "================================================================================"

echo "Step 1.1: Technical Features..."
python src_clean/data_pipelines/silver/market_technical_processor.py \
  --input $BRONZE_MARKET \
  --output $OUTPUT_DIR/silver/market/technical/${INSTRUMENT}_technical.csv

echo "Step 1.2: Microstructure Features..."
python src_clean/data_pipelines/silver/market_microstructure_processor.py \
  --input $BRONZE_MARKET \
  --output $OUTPUT_DIR/silver/market/microstructure/${INSTRUMENT}_microstructure.csv

echo "Step 1.3: Volatility Features..."
python src_clean/data_pipelines/silver/market_volatility_processor.py \
  --input $BRONZE_MARKET \
  --output $OUTPUT_DIR/silver/market/volatility/${INSTRUMENT}_volatility.csv

echo "✓ Market Silver layer complete"
echo ""

# ============================================================================
# STAGE 2: BRONZE → SILVER (NEWS)
# ============================================================================
echo "================================================================================"
echo "STAGE 2: BRONZE → SILVER (NEWS)"
echo "================================================================================"

echo "Step 2.1: Sentiment Features..."
python src_clean/data_pipelines/silver/news_sentiment_processor.py \
  --input-dir $BRONZE_NEWS \
  --output $OUTPUT_DIR/silver/news/sentiment/${INSTRUMENT}_sentiment.csv

echo "✓ News Silver layer complete"
echo ""

# ============================================================================
# STAGE 3: SILVER → GOLD (MARKET)
# ============================================================================
echo "================================================================================"
echo "STAGE 3: SILVER → GOLD (MARKET)"
echo "================================================================================"

echo "Step 3.1: Market Gold Features..."
python src_clean/data_pipelines/gold/market_gold_builder.py \
  --technical $OUTPUT_DIR/silver/market/technical/${INSTRUMENT}_technical.csv \
  --microstructure $OUTPUT_DIR/silver/market/microstructure/${INSTRUMENT}_microstructure.csv \
  --volatility $OUTPUT_DIR/silver/market/volatility/${INSTRUMENT}_volatility.csv \
  --output $OUTPUT_DIR/gold/market/features/${INSTRUMENT}_features.csv

echo "✓ Market Gold layer complete"
echo ""

# ============================================================================
# STAGE 4: GENERATE LABELS
# ============================================================================
echo "================================================================================"
echo "STAGE 4: GENERATE LABELS"
echo "================================================================================"

echo "Step 4.1: Prediction Labels (30-minute)..."
python src_clean/data_pipelines/gold/label_generator.py \
  --input $OUTPUT_DIR/gold/market/features/${INSTRUMENT}_features.csv \
  --output $OUTPUT_DIR/gold/market/labels/${INSTRUMENT}_labels_30min.csv \
  --horizon 30

echo "✓ Labels generated"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "================================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "Data processed:"
ls -lh $OUTPUT_DIR/gold/market/features/ | tail -1
ls -lh $OUTPUT_DIR/gold/market/labels/ | tail -1
echo ""
echo "Next steps:"
echo "  1. Train models:"
echo "     python src_clean/training/xgboost_training_pipeline_mlflow.py \\"
echo "       --market-features $OUTPUT_DIR/gold/market/features/${INSTRUMENT}_features.csv \\"
echo "       --task classification"
echo ""
echo "  2. Select best model:"
echo "     python src_clean/training/model_selector.py --promote"
echo ""
echo "================================================================================"
