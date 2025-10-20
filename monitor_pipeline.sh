#!/bin/bash
# Pipeline monitoring script
# Checks progress every 10 minutes and reports status

LOG_FILE="/tmp/pipeline_5year.log"
MONITOR_LOG="/tmp/pipeline_monitor.log"

echo "====================" >> $MONITOR_LOG
echo "Monitor started: $(date)" >> $MONITOR_LOG
echo "====================" >> $MONITOR_LOG

while true; do
    echo "" >> $MONITOR_LOG
    echo "Check at: $(date)" >> $MONITOR_LOG

    # Check if FinBERT stage is running
    if grep -q "STAGE 4.*NEWS" $LOG_FILE; then
        if grep -q "articles processed" $LOG_FILE 2>/dev/null; then
            PROCESSED=$(grep "articles processed" $LOG_FILE | tail -1)
            echo "FinBERT: $PROCESSED" >> $MONITOR_LOG
        else
            echo "FinBERT: Processing..." >> $MONITOR_LOG
        fi
    fi

    # Check if training started
    if grep -q "STAGE 6.*TRAIN" $LOG_FILE; then
        echo "STATUS: Training model..." >> $MONITOR_LOG
    elif grep -q "STAGE 5.*LABELS" $LOG_FILE; then
        echo "STATUS: Generating labels..." >> $MONITOR_LOG
    elif grep -q "STAGE 4" $LOG_FILE; then
        echo "STATUS: FinBERT processing (longest stage)..." >> $MONITOR_LOG
    fi

    # Check if completed
    if grep -q "PIPELINE COMPLETE" $LOG_FILE; then
        echo "✓ PIPELINE COMPLETE!" >> $MONITOR_LOG
        break
    fi

    # Check for errors
    if grep -q "failed" $LOG_FILE; then
        echo "✗ ERROR DETECTED" >> $MONITOR_LOG
        grep "failed\|error" $LOG_FILE | tail -5 >> $MONITOR_LOG
    fi

    sleep 600  # Wait 10 minutes
done

echo "Monitor ended: $(date)" >> $MONITOR_LOG
