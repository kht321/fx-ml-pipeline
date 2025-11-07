#!/bin/bash

RUN_ID="manual__2025-11-01T19:36:59+00:00"

echo "================================================================================"
echo "📊 MONITORING OPTIMIZED PIPELINE RUN"
echo "================================================================================"
echo "Run ID: $RUN_ID"
echo "Started: 2025-11-01 19:37:00 UTC"
echo "Expected completion: ~25-35 minutes (20:00-20:10 UTC)"
echo ""
echo "FinBERT Optimization: Batch processing (64 articles at once)"
echo "Expected speedup: 20-30x faster than previous sequential processing"
echo "================================================================================"
echo ""

for i in {1..40}; do
    echo "=== Check $i at $(date '+%Y-%m-%d %H:%M:%S') ==="
    
    # Get task states
    STATES=$(docker exec ml-airflow-scheduler airflow tasks states-for-dag-run sp500_ml_pipeline_v4_docker $RUN_ID 2>&1)
    
    # Count tasks by state
    SUCCESS_COUNT=$(echo "$STATES" | grep -c "success" || echo "0")
    RUNNING_COUNT=$(echo "$STATES" | grep -c "running" || echo "0")
    FAILED_COUNT=$(echo "$STATES" | grep -c "failed" || echo "0")
    NONE_COUNT=$(echo "$STATES" | grep -c "None" || echo "0")
    
    echo "✅ Completed: $SUCCESS_COUNT"
    echo "🔄 Running: $RUNNING_COUNT"
    echo "❌ Failed: $FAILED_COUNT"
    echo "⏳ Pending: $NONE_COUNT"
    echo ""
    
    # Show currently running tasks
    if [ "$RUNNING_COUNT" -gt 0 ]; then
        echo "Currently running tasks:"
        echo "$STATES" | grep "running" | awk '{print "  - " $5}' | head -5
        echo ""
    fi
    
    # Check if completed
    if [ "$RUNNING_COUNT" -eq 0 ] && [ "$NONE_COUNT" -eq 0 ]; then
        echo "================================================================================"
        if [ "$FAILED_COUNT" -gt 0 ]; then
            echo "❌ PIPELINE COMPLETED WITH FAILURES"
            echo ""
            echo "Failed tasks:"
            echo "$STATES" | grep "failed" | awk '{print "  - " $5}'
        else
            echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
            echo ""
            echo "All $SUCCESS_COUNT tasks completed successfully"
        fi
        echo "================================================================================"
        break
    fi
    
    sleep 45
done
