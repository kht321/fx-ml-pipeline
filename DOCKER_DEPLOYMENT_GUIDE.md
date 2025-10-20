# Docker Deployment Guide - FinBERT Integration

**Date**: 2025-10-21
**Version**: 2.0 (with FinBERT)
**Status**: ✅ Ready to Deploy

---

## Quick Start

### 1. Rebuild Docker Images

```bash
# Navigate to project root
cd /path/to/fx-ml-pipeline

# Rebuild all images with FinBERT
docker-compose -f docker/docker-compose.yml build --no-cache

# Or rebuild specific services
docker-compose -f docker/docker-compose.yml build dev
docker-compose -f docker/docker-compose.yml build api
docker-compose -f docker/docker-compose.yml build streamlit
```

**Expected build time**: 15-20 minutes (first time, with FinBERT download)

---

### 2. Rebuild Airflow Containers

```bash
# Navigate to Airflow directory
cd airflow_mlops

# Rebuild Airflow images
docker-compose build --no-cache

# Start Airflow services
docker-compose up -d

# Check status
docker-compose ps
```

---

### 3. Verify FinBERT Installation

```bash
# Test in dev container
docker-compose -f docker/docker-compose.yml run --rm dev python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print('Testing FinBERT installation...')
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
print('✓ FinBERT loaded successfully')
print(f'Model size: {sum(p.numel() for p in model.parameters())} parameters')
print(f'Device: {\"GPU\" if torch.cuda.is_available() else \"CPU\"}')
"
```

**Expected output**:
```
Testing FinBERT installation...
✓ FinBERT loaded successfully
Model size: 109483779 parameters
Device: CPU
```

---

## Detailed Deployment Steps

### Step 1: Pre-Deployment Checklist

- [ ] Git pull latest changes
- [ ] Check disk space (need ~10 GB for Docker images)
- [ ] Check memory (need 4-8 GB available)
- [ ] Backup existing models (if any)
- [ ] Review `requirements.txt` changes

```bash
# Check disk space
df -h

# Check Docker disk usage
docker system df

# Clean up old images (optional)
docker system prune -a
```

---

### Step 2: Main Application Deployment

#### A. Development Environment

```bash
cd docker

# Build development image
docker-compose build dev

# Run development container
docker-compose run --rm dev bash

# Inside container, test pipeline
python src_clean/run_full_pipeline.py \
    --bronze-market data/bronze/market/spx500_usd_m1_5years.ndjson \
    --bronze-news data/bronze/news/historical_5year \
    --output-dir data_output \
    --instrument spx500
```

---

#### B. API Service

```bash
# Build API image
docker-compose build api

# Start API service
docker-compose up -d api

# Test API health
curl http://localhost:8000/health

# Test FinBERT endpoint (if exposed)
curl http://localhost:8000/api/sentiment -X POST \
    -H "Content-Type: application/json" \
    -d '{"text": "Stock market rallies on strong earnings"}'
```

---

#### C. Streamlit Dashboard

```bash
# Build Streamlit image
docker-compose build streamlit

# Start Streamlit service
docker-compose up -d streamlit

# Access dashboard
open http://localhost:8501
```

---

### Step 3: Airflow MLOps Deployment

#### A. Update Airflow Environment

```bash
cd airflow_mlops

# Build Airflow images (includes FinBERT in ETL container)
docker-compose build --no-cache

# Start Airflow stack
docker-compose up -d

# Wait for services to be healthy (30-60 seconds)
docker-compose ps
```

---

#### B. Deploy New DAG

```bash
# The new DAG is automatically detected
# Check Airflow UI: http://localhost:8080
# Login: admin / admin

# Enable the new DAG
airflow dags unpause sp500_ml_pipeline_finbert

# Trigger manually (first run)
airflow dags trigger sp500_ml_pipeline_finbert

# Monitor execution
airflow dags list-runs -d sp500_ml_pipeline_finbert
```

---

#### C. Verify Airflow ETL Container

```bash
# Check ETL container has FinBERT
docker-compose run --rm etl python -c "
import torch
import transformers
print(f'torch: {torch.__version__}')
print(f'transformers: {transformers.__version__}')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
print('✓ FinBERT available in ETL container')
"
```

---

### Step 4: Production Deployment

#### A. Environment Variables

```bash
# Create production .env file
cat > .env << EOF
# OANDA API
OANDA_TOKEN=your_production_token
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENV=practice  # or 'live' for production

# News APIs (optional)
NEWSAPI_KEY=your_key
ALPHAVANTAGE_KEY=your_key
FINNHUB_KEY=your_key

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5002

# FinBERT Configuration
FINBERT_DEVICE=cpu  # or 'cuda' if GPU available
FINBERT_BATCH_SIZE=32
EOF
```

---

#### B. Full Stack Deployment

```bash
# Use full-stack compose file
docker-compose -f docker/docker-compose.full-stack.yml up -d

# Services started:
# - FastAPI (8000)
# - Streamlit (8501)
# - MLflow (5002)
# - Evidently (8050)
# - Postgres (5432)
# - Redis (6379)

# Check all services
docker-compose -f docker/docker-compose.full-stack.yml ps
```

---

### Step 5: Monitoring & Validation

#### A. Check Logs

```bash
# Check API logs
docker-compose logs -f api

# Check Airflow scheduler logs
cd airflow_mlops
docker-compose logs -f scheduler

# Check specific task logs (FinBERT)
docker-compose logs etl | grep -i finbert
```

---

#### B. Monitor Resource Usage

```bash
# Monitor Docker container resources
docker stats

# Expected for FinBERT processing:
# - Memory: 2-3 GB during news gold layer
# - CPU: 80-100% during processing
# - Disk I/O: Moderate

# Check specific container
docker stats airflow_mlops_etl_1
```

---

#### C. Validate Pipeline Output

```bash
# Check gold layer output
docker-compose -f docker/docker-compose.yml run --rm dev bash -c "
ls -lh /app/data/gold/news/signals/
head -5 /app/data/gold/news/signals/spx500_news_signals.csv
wc -l /app/data/gold/news/signals/spx500_news_signals.csv
"

# Should show:
# - CSV file with trading signals
# - Columns: signal_time, avg_sentiment, signal_strength, trading_signal, etc.
# - Multiple thousand rows (depends on data)
```

---

## Troubleshooting

### Issue 1: FinBERT Download Fails

**Symptom**: "ConnectionError" or "TimeoutError" during build

**Solution**:
```bash
# Pre-download model outside Docker
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained('ProsusAI/finbert')
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
"

# Copy to Docker cache
docker cp ~/.cache/huggingface $(docker ps -q -f name=dev):/root/.cache/

# Or rebuild with network timeout increased
docker-compose build --build-arg NETWORK_TIMEOUT=300 dev
```

---

### Issue 2: Out of Memory

**Symptom**: Container crashes during FinBERT processing

**Solution**:
```bash
# Increase Docker memory limit
# Docker Desktop -> Settings -> Resources -> Memory
# Recommended: 8 GB minimum

# Or process in smaller batches
# Edit airflow_mlops/dags/sp500_ml_pipeline_finbert.py
# Add: --batch-size 16  # default is 32
```

---

### Issue 3: Airflow DAG Not Appearing

**Symptom**: New DAG doesn't show in Airflow UI

**Solution**:
```bash
# Restart Airflow scheduler
cd airflow_mlops
docker-compose restart scheduler

# Check DAG for syntax errors
docker-compose exec scheduler airflow dags list | grep finbert

# Check DAG import errors
docker-compose exec scheduler airflow dags list-import-errors
```

---

### Issue 4: FinBERT Processing Too Slow

**Symptom**: News gold layer takes >1 hour

**Solutions**:
1. **Enable GPU** (if available):
   ```dockerfile
   # In docker/Dockerfile
   FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
   ```

2. **Reduce article count**:
   ```python
   # Process only recent articles
   --filter-days 30  # Only last 30 days
   ```

3. **Increase batch size** (if memory allows):
   ```bash
   # In news_signal_builder.py, modify batch processing
   ```

---

## Performance Benchmarks

### Build Times

| Image | Without FinBERT | With FinBERT | Increase |
|-------|-----------------|--------------|----------|
| Dev | 5 min | 15 min | +10 min |
| API | 4 min | 12 min | +8 min |
| Streamlit | 5 min | 13 min | +8 min |
| Airflow ETL | 2 min | 10 min | +8 min |

### Runtime Performance

| Task | CPU | GPU | Notes |
|------|-----|-----|-------|
| FinBERT (1K articles) | 1-2 min | 5-10 sec | 50-100x faster with GPU |
| Full pipeline (12K articles) | 15-20 min | 2-3 min | Dominated by FinBERT |
| Training | 30-60 sec | 30-60 sec | XGBoost uses CPU |

### Resource Usage

| Service | Memory | Disk | Notes |
|---------|--------|------|-------|
| Dev Container | 1-3 GB | 2.5 GB | Includes FinBERT model |
| ETL Container | 2-4 GB | 1.7 GB | During FinBERT processing |
| API Service | 800 MB | 2.3 GB | Idle (no FinBERT loaded) |
| Airflow Stack | 2-4 GB | 5 GB | All services |

---

## Rollback Procedure

If you need to rollback to pre-FinBERT version:

```bash
# 1. Checkout previous commit
git checkout <commit-before-finbert>

# 2. Rebuild images
docker-compose build --no-cache

# 3. Restart services
docker-compose down
docker-compose up -d

# 4. Verify old pipeline works
docker-compose run --rm dev python src_clean/run_full_pipeline.py --skip-news
```

---

## Next Steps

### After Successful Deployment:

1. **Monitor First Production Run**
   ```bash
   # Watch Airflow DAG execution
   # Check logs for FinBERT processing
   # Verify model training completes
   ```

2. **Performance Tuning**
   - Adjust `FINBERT_WINDOW` based on signal quality
   - Experiment with batch sizes
   - Consider GPU upgrade for faster processing

3. **MLflow Integration**
   - Track experiments with/without news features
   - Compare model performance
   - Monitor feature importance

4. **Set Up Alerts**
   - Airflow task failures
   - Memory usage warnings
   - FinBERT processing timeouts

---

## Support & Documentation

- **FinBERT Implementation**: `FINBERT_IMPLEMENTATION_SUMMARY.md`
- **Infrastructure Updates**: `INFRASTRUCTURE_UPDATES_FINBERT.md`
- **Test Results**: `TEST_RESULTS.md`
- **Main README**: `README.md`
- **Airflow DAG**: `airflow_mlops/dags/sp500_ml_pipeline_finbert.py`

---

## Summary Checklist

Deployment steps:

- [ ] Git pull latest changes
- [ ] Review `requirements.txt` (torch, transformers added)
- [ ] Rebuild Docker images (15-20 min first time)
- [ ] Rebuild Airflow containers (10-15 min)
- [ ] Test FinBERT in dev container
- [ ] Verify Airflow DAG appears
- [ ] Trigger test run
- [ ] Monitor logs for errors
- [ ] Validate output files
- [ ] Check model performance

**Estimated total deployment time**: 60-90 minutes

**Status**: ✅ **Ready for Production Deployment**

---

**Last Updated**: 2025-10-21
**Maintainer**: ML Team
**Version**: 2.0 (FinBERT Integration)
