# Multi-stage Dockerfile for S&P 500 ML Pipeline
# Optimized for development and production use

# ============================================================================
# Stage 1: Base Image with System Dependencies
# ============================================================================
FROM python:3.11-slim as base

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
    pip install --no-cache-dir -r requirements.txt


# ============================================================================
# Stage Experiment
# ============================================================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as gpu-base

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    python3.11-venv \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create python symlinks
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Set CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src_clean/ /app/src_clean/
COPY scripts/ /app/scripts/
COPY notebooks/ /app/notebooks/
COPY configs/ /app/configs/
COPY *.py /app/

EXPOSE 8888 8000 5000 8501
CMD ["bash"]

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

# Set Python path
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8888 8000 5000 8501

# Set default command
CMD ["/bin/bash"]

# ============================================================================
# Stage 4: Production Image (minimal, optimized)
# ============================================================================
FROM base as production

# Copy only necessary files from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY src_clean/ /app/src_clean/
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


# ============================================================================
# Stage 7: Streamlit App Image (for UI)
# ============================================================================
FROM production as streamlit
RUN pip install \
    streamlit \
    plotly \
    oandapyV20

EXPOSE 8501

CMD ["streamlit", "run", "src/ui_streamlit.py", "--server.port=8501", "--server.address=http://localhost"]
# ============================================================================
