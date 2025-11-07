#!/bin/bash
# Start all components of the ML Pipeline

echo "=========================================="
echo "Starting S&P 500 ML Pipeline Services"
echo "=========================================="
echo ""

# Function to check if port is in use
check_port() {
    lsof -ti:$1 > /dev/null 2>&1
    return $?
}

# Kill existing processes
echo "Cleaning up existing processes..."
pkill -f "news-simulator/app.py" 2>/dev/null
pkill -f "uvicorn src_clean.api.main" 2>/dev/null
pkill -f "streamlit run" 2>/dev/null
sleep 2

# Start News Simulator (Port 5000)
echo ""
echo "1. Starting News Simulator on port 5000..."
cd docker/tools/news-simulator
export NEWS_OUTPUT_DIR="../../../data_clean/bronze/news/simulated"
../../../.venv/bin/python3 app.py > ../../../logs/news_simulator.log 2>&1 &
NEWS_PID=$!
cd ../../..
sleep 3

if check_port 5000; then
    echo "   ✓ News Simulator running (PID: $NEWS_PID)"
    echo "   → Web UI: http://localhost:5000"
else
    echo "   ✗ Failed to start News Simulator"
fi

# Start FastAPI (Port 8000)
echo ""
echo "2. Starting FastAPI Prediction Service on port 8000..."
.venv/bin/uvicorn src_clean.api.main:app --host 0.0.0.0 --port 8000 > logs/fastapi.log 2>&1 &
API_PID=$!
sleep 3

if check_port 8000; then
    echo "   ✓ FastAPI running (PID: $API_PID)"
    echo "   → API Docs: http://localhost:8000/docs"
    echo "   → Health: http://localhost:8000/health"
else
    echo "   ✗ Failed to start FastAPI"
fi

# Start Streamlit (Port 8501)
echo ""
echo "3. Starting Streamlit Dashboard on port 8501..."
.venv/bin/streamlit run src_clean/ui/streamlit_dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!
sleep 5

if check_port 8501; then
    echo "   ✓ Streamlit running (PID: $STREAMLIT_PID)"
    echo "   → Dashboard: http://localhost:8501"
else
    echo "   ✗ Failed to start Streamlit"
fi

# Summary
echo ""
echo "=========================================="
echo "All Services Started!"
echo "=========================================="
echo ""
echo "Services:"
echo "  • News Simulator:  http://localhost:5000"
echo "  • FastAPI:         http://localhost:8000/docs"
echo "  • Streamlit:       http://localhost:8501"
echo ""
echo "Logs:"
echo "  • News Simulator:  logs/news_simulator.log"
echo "  • FastAPI:         logs/fastapi.log"
echo "  • Streamlit:       logs/streamlit.log"
echo ""
echo "To stop all services:"
echo "  ./stop_all.sh"
echo ""
echo "=========================================="
