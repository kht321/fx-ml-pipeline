# Streamlit Dashboard Dockerfile
# Python 3.11 with pinned versions

FROM python:3.11.13-slim

LABEL maintainer="fx-ml-pipeline"
LABEL python.version="3.11.13"
LABEL description="Streamlit dashboard for S&P 500 ML predictions"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies with pinned versions
RUN pip install --no-cache-dir --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./src_clean/ui /app/src_clean/ui
COPY ./src_clean/__init__.py /app/src_clean/

# Create mount points for runtime model artifacts
RUN mkdir -p /app/data_clean/models

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set Python path
ENV PYTHONPATH=/app

# Run Streamlit
CMD ["streamlit", "run", "src_clean/ui/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
