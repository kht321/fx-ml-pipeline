#!/bin/bash
# Start the Streamlit Dashboard

# Kill any existing Streamlit process
pkill -f "streamlit run" 2>/dev/null

# Start Streamlit
.venv/bin/streamlit run src_clean/ui/streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
