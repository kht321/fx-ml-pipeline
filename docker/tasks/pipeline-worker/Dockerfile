# Pipeline Worker Dockerfile
# Unified image for all pipeline tasks: data collection, feature engineering, training
# Python 3.11 with full dependencies from requirements.txt

FROM python:3.11.13-slim

LABEL maintainer="fx-ml-pipeline"
LABEL python.version="3.11.13"
LABEL description="Pipeline worker for data collection, feature engineering, and model training"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies with pinned versions
RUN pip install --no-cache-dir --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire src_clean codebase
COPY ./src_clean /app/src_clean

# Create __init__.py if not exists
RUN touch /app/src_clean/__init__.py

# Set Python path
ENV PYTHONPATH=/app

# Default entrypoint - can be overridden by DockerOperator
ENTRYPOINT ["python"]
