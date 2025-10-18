# Multi-stage Dockerfile for S&P 500 ML Pipeline
# Optimized for development and production use

# ============================================================================
# Stage 1: Base Image with System Dependencies
# ============================================================================
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================================================
# Stage 3: Development Image (with dev tools)
# ============================================================================
FROM dependencies as development

# Install development tools
RUN pip install \
    jupyter \
    ipython \
    pytest \
    black \
    flake8 \
    pylint

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p \
    data/bronze/prices \
    data/news/bronze/raw_articles \
    data/news/silver/sentiment_scores \
    data/news/silver/entity_mentions \
    data/news/silver/topic_signals \
    data/news/gold/news_signals \
    data/sp500/bronze/prices \
    data/sp500/silver/technical_features \
    data/sp500/silver/microstructure \
    data/sp500/silver/volatility \
    data/sp500/gold/training \
    outputs \
    models

# Expose ports
EXPOSE 8888 8000 5000

# Set default command
CMD ["/bin/bash"]

# ============================================================================
# Stage 4: Production Image (minimal, optimized)
# ============================================================================
FROM base as production

# Copy only necessary files from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY *.py /app/
COPY requirements.txt /app/

# Create data directories
RUN mkdir -p \
    /app/data/bronze/prices \
    /app/data/news \
    /app/data/sp500 \
    /app/outputs \
    /app/models

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Set default command
CMD ["python", "--version"]

# ============================================================================
# Stage 5: Jupyter Notebook Image
# ============================================================================
FROM development as jupyter

# Install Jupyter extensions
RUN pip install \
    jupyterlab \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly

# Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ============================================================================
# Stage 6: API Server Image (for model serving)
# ============================================================================
FROM production as api

# Install FastAPI and serving dependencies
USER root
RUN pip install \
    fastapi \
    uvicorn[standard] \
    pydantic

USER mluser

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
