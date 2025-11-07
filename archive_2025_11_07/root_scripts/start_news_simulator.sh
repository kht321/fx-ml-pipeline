#!/bin/bash
# Start the News Simulator

# Kill any existing process on port 5000
lsof -ti:5000 | xargs kill -9 2>/dev/null

# Navigate to the news simulator directory
cd docker/tools/news-simulator

# Set output directory
export NEWS_OUTPUT_DIR="../../../data_clean/bronze/news/simulated"

# Start the Flask app
../../../.venv/bin/python3 app.py

