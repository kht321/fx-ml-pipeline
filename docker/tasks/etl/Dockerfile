FROM python:3.11-slim
WORKDIR /app

# Install FinBERT dependencies for news gold layer processing
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    torch==2.2.0 \
    transformers==4.40.0 \
    tqdm==4.67.1

# Pre-download FinBERT model to avoid runtime download
# This adds ~440MB to the image but eliminates first-run latency
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
    AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')" || \
    echo "Warning: FinBERT pre-download failed, will download at runtime"

COPY docker/tasks/etl/steps /app/steps
ENTRYPOINT ["python"]
