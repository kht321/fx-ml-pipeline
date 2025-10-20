# Infrastructure Updates for FinBERT Implementation

**Date**: 2025-10-21
**Changes**: FinBERT gold layer integration
**Status**: ✅ Required updates identified

---

## Summary

The FinBERT implementation adds new dependencies (`torch`, `transformers`) that need to be included in Docker, Airflow, and MLflow environments.

---

## 1. Docker Updates Required

### 📝 **Files to Update**:

#### A. Main Dockerfile (`docker/Dockerfile`)

**Current**: Python 3.9, no torch/transformers
**Required**: Add FinBERT dependencies

**Update Line 37** (in dependencies stage):
```dockerfile
# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt
```

**Action**: ✅ Already uses `requirements.txt` - **NO CHANGE NEEDED**
- The Dockerfile installs from `requirements.txt`
- We already updated `requirements.txt` with torch and transformers
- **Docker will automatically pick up the new dependencies on rebuild**

**Recommendation**: Add FinBERT model download to reduce first-run time

**Add after line 37**:
```dockerfile
# Pre-download FinBERT model to avoid download during runtime
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
    AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"
```

**Impact**: Adds ~440MB to Docker image, but eliminates download wait at runtime

---

#### B. Docker Compose (`docker/docker-compose.yml`)

**Status**: ✅ **NO CHANGES NEEDED**
- Already mounts volumes correctly
- Environment variables already configured
- Ports and networking correct

---

#### C. API/Streamlit Dockerfiles

**Files**:
- `docker/Dockerfile.fastapi`
- `docker/Dockerfile.streamlit`

**Status**: ⚠️ **CHECK IF THEY USE requirements.txt**

If they have hardcoded dependencies, need to add:
```dockerfile
RUN pip install torch==2.2.0 transformers==4.40.0
```

---

## 2. Airflow Updates Required

### 📝 **Files to Update**:

#### A. Airflow Main Dockerfile (`airflow_mlops/Dockerfile`)

**Check**: Does it use the project's requirements.txt?

**If NO**: Add to Dockerfile:
```dockerfile
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    transformers==4.40.0
```

#### B. ETL Docker Container (`airflow_mlops/docker/etl/Dockerfile`)

**Current**:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas==2.2.2
COPY steps /app/steps
ENTRYPOINT ["python"]
```

**Status**: ⚠️ **NEEDS UPDATE IF ETL RUNS NEWS GOLD LAYER**

**Add**:
```dockerfile
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    torch==2.2.0 \
    transformers==4.40.0 \
    tqdm==4.67.1
```

**Impact**: Container will grow from ~200MB to ~1.5GB (due to torch)

---

#### C. Trainer Docker Container (`airflow_mlops/docker/trainer/Dockerfile`)

**Current**:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas==2.2.2 scikit-learn==1.5.1
COPY train.py /app/train.py
ENTRYPOINT ["python","/app/train.py"]
```

**Status**: ✅ **PROBABLY NO CHANGE NEEDED**
- Training reads pre-computed gold layer signals
- Doesn't run FinBERT during training
- **Only needs to read CSV files**

---

#### D. Airflow DAGs

**Files to Check**:
- `airflow_mlops/dags/data_pipeline.py`
- `airflow_mlops/dags/train_deploy_pipeline.py`

**Current**: Demo pipeline with dummy data

**Status**: ⚠️ **NEEDS UPDATE TO CALL NEW GOLD LAYER**

**Required Changes**:

1. **Add new task for FinBERT gold layer**:

```python
from airflow.operators.bash import BashOperator

build_news_gold = BashOperator(
    task_id='build_news_gold',
    bash_command='''
        python /app/src_clean/data_pipelines/gold/news_signal_builder.py \
            --silver-sentiment /opt/airflow/data/silver/news/sentiment/spx500_sentiment.csv \
            --bronze-news /opt/airflow/data/bronze/news \
            --output /opt/airflow/data/gold/news/signals/spx500_news_signals.csv \
            --window 60
    ''',
    dag=dag
)
```

2. **Update task dependencies**:

```python
# Old flow
ingest >> validate >> transform >> build_features >> train

# New flow with FinBERT
ingest >> validate >> transform >> [build_market_gold, build_news_gold] >> train
```

---

## 3. MLflow Updates

### 📝 **Configuration**:

**Status**: ✅ **NO CHANGES NEEDED**

MLflow tracking:
- ✅ Tracks experiments automatically
- ✅ Logs model with news features
- ✅ Records feature names (including `news_*` columns)
- ✅ Already integrated in `xgboost_training_pipeline_mlflow.py`

**What MLflow Will Track**:
- Model trained with news features
- Feature importance (including FinBERT features)
- Metrics with/without news signals
- Hyperparameters

**No code changes needed** - MLflow auto-detects features from training data

---

## 4. Production Deployment Checklist

### Before Deployment:

- [ ] **Update Docker images**:
  ```bash
  docker-compose build --no-cache
  ```

- [ ] **Test FinBERT in container**:
  ```bash
  docker-compose run dev python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ProsusAI/finbert')"
  ```

- [ ] **Update Airflow DAGs** with new gold layer task

- [ ] **Verify Airflow containers have torch/transformers**:
  ```bash
  docker exec airflow-webserver pip list | grep -E "torch|transformers"
  ```

- [ ] **Test full pipeline in Docker**:
  ```bash
  docker-compose run dev python src_clean/run_full_pipeline.py \
      --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
      --bronze-news data_clean/bronze/news/historical_5year \
      --output-dir data_clean_5year
  ```

- [ ] **Check MLflow experiment tracking**:
  - Verify new experiments appear in MLflow UI
  - Check that `news_*` features are logged
  - Verify model artifacts include FinBERT outputs

---

## 5. Resource Requirements

### Updated Container Sizes:

| Container | Before | After (with FinBERT) | Increase |
|-----------|--------|---------------------|----------|
| **Dev** | ~800 MB | ~2.5 GB | +1.7 GB |
| **ETL** | ~200 MB | ~1.7 GB | +1.5 GB |
| **Trainer** | ~300 MB | ~300 MB | No change |
| **API** | ~600 MB | ~2.3 GB | +1.7 GB |
| **Streamlit** | ~700 MB | ~2.4 GB | +1.7 GB |

### Runtime Memory:

| Service | Before | After (with FinBERT) |
|---------|--------|---------------------|
| ETL Container | 500 MB | 1.5 GB |
| News Gold Task | N/A | **2-3 GB** (FinBERT processing) |
| Training | 2 GB | 2 GB (reads pre-computed signals) |

### Disk Space:

- Docker images: +~5 GB (across all services)
- FinBERT model: +440 MB (cached once)
- Total: **~5.5 GB additional**

---

## 6. Performance Optimizations

### For Production:

1. **Pre-download models in Dockerfile**:
   ```dockerfile
   RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('ProsusAI/finbert'); AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"
   ```
   - Benefit: No download delay at runtime
   - Cost: +440MB image size

2. **GPU Support (Optional)**:
   ```dockerfile
   # Use CUDA base image for GPU acceleration
   FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
   ```
   - Benefit: 50-100x faster FinBERT processing
   - Requires: NVIDIA GPU with Docker GPU support

3. **Multi-stage Docker build**:
   ```dockerfile
   # Build stage: Install all deps
   FROM python:3.11-slim as builder
   RUN pip install torch transformers --target /install

   # Runtime stage: Copy only built packages
   FROM python:3.11-slim
   COPY --from=builder /install /usr/local/lib/python3.11/site-packages
   ```
   - Benefit: Smaller final image
   - Complexity: Higher

---

## 7. Minimal Update Path (Quick Deploy)

**If you want to deploy quickly with minimal changes**:

### Option A: Update requirements.txt only

```bash
# 1. Rebuild all images
docker-compose build --no-cache

# 2. Restart services
docker-compose down
docker-compose up -d
```

**Result**: All services get torch+transformers automatically

**Downside**: Larger images, longer build time

---

### Option B: Selective updates (Recommended)

**Only update containers that process news**:

1. **ETL container**: Add torch+transformers (needs FinBERT)
2. **Dev container**: Already uses requirements.txt (auto-updated)
3. **Trainer container**: NO CHANGE (reads pre-computed signals)

**Result**: Minimal image size increase, faster deployment

---

## 8. Testing Commands

### Test Docker Build:
```bash
cd docker
docker build -t sp500-finbert:test --target production .
docker run --rm sp500-finbert:test python -c "import torch; import transformers; print('FinBERT deps OK')"
```

### Test Airflow ETL Container:
```bash
cd airflow_mlops/docker/etl
docker build -t airflow-etl-finbert:test .
docker run --rm airflow-etl-finbert:test python -c "from transformers import AutoTokenizer; print('OK')"
```

### Test Full Pipeline in Docker:
```bash
docker-compose run --rm dev bash -c "\
  python src_clean/data_pipelines/gold/news_signal_builder.py \
    --silver-sentiment data_clean/silver/news/sentiment/sp500_sentiment.csv \
    --bronze-news data_clean/bronze/news \
    --output /tmp/test_signals.csv \
    --window 60 && \
  echo 'Pipeline test: SUCCESS'"
```

---

## 9. Rollback Plan

If FinBERT causes issues:

```bash
# 1. Revert requirements.txt
git checkout HEAD~1 requirements.txt

# 2. Rebuild without FinBERT
docker-compose build --no-cache

# 3. Use old pipeline (skip Stage 4)
python src_clean/run_full_pipeline.py \
    --skip-news \
    # ... other args
```

---

## 10. Summary

### ✅ **Ready to Deploy**:
- requirements.txt already updated
- Main codebase ready
- Tests passing

### ⚠️ **Needs Updates**:
1. **Airflow ETL Dockerfile**: Add torch+transformers
2. **Airflow DAG**: Add news gold layer task
3. **Docker images**: Rebuild with new requirements

### 📝 **Recommended Next Steps**:

1. **Test locally first**:
   ```bash
   docker-compose build dev
   docker-compose run dev python tests/test_finbert_gold_layer.py
   ```

2. **Update Airflow ETL container**:
   - Edit `airflow_mlops/docker/etl/Dockerfile`
   - Add torch+transformers dependencies

3. **Update Airflow DAG**:
   - Add `build_news_gold` task
   - Update task dependencies

4. **Rebuild and deploy**:
   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```

5. **Monitor first run**:
   - Check Docker logs for FinBERT model download
   - Verify MLflow tracks experiments correctly
   - Test with small dataset first

---

## 11. Estimated Deployment Time

| Task | Time | Notes |
|------|------|-------|
| Update Dockerfiles | 15 min | Edit 2-3 files |
| Update Airflow DAG | 15 min | Add 1 task |
| Build Docker images | 30-45 min | With FinBERT download |
| Test in container | 15 min | Run test suite |
| Deploy to production | 10 min | docker-compose up |
| **Total** | **~90 min** | **Plus monitoring time** |

---

**Status**: 📋 **Implementation guide ready**
**Next**: Update Dockerfiles and DAGs, then deploy
**Risk**: Low (backward compatible, can rollback easily)
