#!/bin/bash

echo "=== Drift Detection Simulation Test ==="
echo ""
echo "Step 1: Stream 10 extremely POSITIVE news articles..."
for i in {1..10}; do
    curl -s -X POST http://localhost:5050/api/stream/custom \
      -H "Content-Type: application/json" \
      -d "{\"text\":\"BREAKING: S&P 500 surges $i% on massive rally, gains surge, bullish growth, positive outlook\"}" > /dev/null
    echo "  ✓ Positive news $i streamed"
    sleep 0.5
done

echo ""
echo "Step 2: Generate predictions with positive sentiment baseline..."
for i in {1..10}; do
    curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"instrument":"SPX500_USD"}' > /dev/null
    echo "  ✓ Prediction $i generated"
    sleep 0.5
done

echo ""
echo "Step 3: Stream 10 extremely NEGATIVE news articles (causing drift)..."
for i in {1..10}; do
    curl -s -X POST http://localhost:5050/api/stream/custom \
      -H "Content-Type: application/json" \
      -d "{\"text\":\"CRASH ALERT: S&P 500 plunges $i%, massive decline, bearish drop, negative fall, crash fears\"}" > /dev/null
    echo "  ✓ Negative news $i streamed"
    sleep 0.5
done

echo ""
echo "Step 4: Generate predictions with negative sentiment (drift expected)..."
for i in {1..10}; do
    curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"instrument":"SPX500_USD"}' > /dev/null
    echo "  ✓ Prediction $i generated"
    sleep 0.5
done

echo ""
echo "Step 5: Check for drift..."
curl -s http://localhost:8000/monitoring/drift/check | jq '.'

echo ""
echo "=== Test Complete ==="
