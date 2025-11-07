#!/bin/bash
# Stop all ML Pipeline services

echo "Stopping all services..."

pkill -f "news-simulator/app.py" && echo "✓ Stopped News Simulator"
pkill -f "uvicorn src_clean.api.main" && echo "✓ Stopped FastAPI"
pkill -f "streamlit run" && echo "✓ Stopped Streamlit"

sleep 2

# Check if anything is still running
if lsof -ti:5000 > /dev/null 2>&1; then
    echo "⚠ Port 5000 still in use"
fi

if lsof -ti:8000 > /dev/null 2>&1; then
    echo "⚠ Port 8000 still in use"
fi

if lsof -ti:8501 > /dev/null 2>&1; then
    echo "⚠ Port 8501 still in use"
fi

echo ""
echo "All services stopped."
