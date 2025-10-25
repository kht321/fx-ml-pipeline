# Codebase Cleanup Summary - October 26, 2025

## Overview

Performed comprehensive codebase cleanup to focus on the clean medallion architecture and remove legacy/unused code.

## Actions Taken

### 1. **Archived Old Code**

Created `archive/unused_code_20251026/` containing:

#### Old Pipeline Scripts
- `run_sp500_pipeline.py` → Replaced by `src_clean/run_full_pipeline.py`
- `monitor_pipeline.sh` → Replaced by MLflow monitoring

#### Migration & Analysis Tools (One-time use)
- `migrate_to_clean_structure.py` - Data migration script (already completed)
- `analyze_data_coverage.py` - Data analysis script (already used)

#### Development/Testing
- `test-streamlit.ipynb` - Old testing notebook

#### Feature Store (Optional - batch focus)
- `feature_repo/` - Feast feature store configuration
  - Only used in `src_clean/api/inference.py`
  - Archived as batch inference is the current priority
  - Can be restored if real-time serving needed

#### Duplicate Airflow Setup
- `airflow_mlops/` - Standalone Airflow configuration
  - Duplicated by `docker/airflow/`
  - Keeping `docker/` as single source of truth

#### Old News Tools
- `news-simulator/` - News article simulator for testing
- `setup_news_api_keys.sh` - Old API key setup script
- `setup_news_cron.sh` - Old cron job setup script

### 2. **Current Active Structure**

```
fx-ml-pipeline/
├── src_clean/                    # ✅ ACTIVE - Clean medallion architecture
│   ├── data_pipelines/
│   │   ├── bronze/              # Raw data ingestion
│   │   ├── silver/              # Feature engineering
│   │   └── gold/                # Training-ready data + labels
│   ├── training/                # XGBoost + MLflow + Model Selector
│   ├── api/                     # REST API (uses Feast - archived but can restore)
│   ├── ui/                      # Streamlit dashboards
│   └── run_full_pipeline.py     # Main orchestrator
│
├── scripts/                      # ✅ ACTIVE - Utility scripts
│   ├── process_5year_data.sh   # 5-year data processing
│   ├── register_model_mlflow.py
│   └── download_sp500_data.sh
│
├── docker/                       # ✅ ACTIVE - Unified Docker setup
│   ├── airflow/                 # Airflow orchestration
│   ├── api/                     # FastAPI service
│   ├── monitoring/              # Grafana, Prometheus
│   ├── tasks/                   # Data quality
│   ├── tools/                   # MLflow
│   └── ui/                      # Streamlit
│
├── docs/                         # ✅ ACTIVE - Documentation
│   ├── ML_PIPELINE_DESIGN.md   # End-to-end ML pipeline design
│   ├── QUICK_START_5YEAR.md    # 5-year data processing guide
│   └── CODEBASE_CLEANUP_20251026.md (this file)
│
├── tests/                        # ✅ ACTIVE - Validation tests
│   ├── test_finbert_gold_layer.py
│   └── validate_implementation.py
│
├── data_clean/                   # ✅ ACTIVE - Production data
├── data_clean_5year/            # ✅ ACTIVE - 5-year historical data
├── mlruns/                      # ✅ ACTIVE - MLflow experiments
│
└── archive/                      # 📦 Archived code
    ├── old_src_20251016/
    ├── old_src_20251018_004940/
    ├── old_docker_20251021/
    ├── old_docker_structure_20251021/
    ├── news_scrapers_20251019/
    ├── usd_sgd_data/
    └── unused_code_20251026/    # ← NEW: This cleanup
```

## Benefits

1. **Cleaner Root Directory**: Removed 7 old scripts and 3 large directories from root
2. **Clear Active Code**: Easy to identify what's actively maintained (`src_clean/`, `scripts/`, `docker/`)
3. **Preserved History**: All old code archived and documented, can be restored if needed
4. **Single Source of Truth**: 
   - Pipeline: `src_clean/run_full_pipeline.py`
   - Docker: `docker/` (removed duplicate `airflow_mlops/`)
5. **Focus on Batch ML**: Archived Feast (real-time serving) to focus on batch inference

## What's Still Active

### Core Data Pipeline (`src_clean/data_pipelines/`)
- **Bronze**: `market_data_downloader.py`, `hybrid_news_scraper.py`
- **Silver**: `market_technical_processor.py`, `market_microstructure_processor.py`, `market_volatility_processor.py`, `news_sentiment_processor.py`
- **Gold**: `market_gold_builder.py`, `news_signal_builder.py` (FinBERT), `label_generator.py`

### ML Training (`src_clean/training/`)
- `xgboost_training_pipeline_mlflow.py` - MLflow version (experiment tracking)
- `xgboost_training_pipeline.py` - Non-MLflow version (orchestrator use)
- `model_selector.py` - Automated model selection (OOT-based)

### UI (`src_clean/ui/`)
- `streamlit_dashboard.py` - Main dashboard
- `ui_streamlit.py` - Alternative dashboard
- `realtime_predictor.py` - Real-time predictions

### API (`src_clean/api/`) - Uses Feast
- `main.py` - FastAPI server
- `inference.py` - Model serving
- `models.py` - Pydantic models

**Note**: API still references `feature_repo/` (now archived). If real-time serving is needed, restore `feature_repo/` from archive.

## Restoration

If any archived files are needed:

```bash
# Restore specific file
cp archive/unused_code_20251026/<filename> .

# Restore Feast for real-time API
cp -r archive/unused_code_20251026/feature_repo .

# Restore Airflow standalone setup
cp -r archive/unused_code_20251026/airflow_mlops .
```

## Next Steps

1. **Update README.md**: Remove references to archived tools
2. **Update docker-compose.yml**: Remove references to `airflow_mlops/` if any
3. **Consider**: Archive `src_clean/api/` if real-time serving not needed
4. **Consider**: Archive `tests/` if validation complete

## Files Preserved (Not Archived)

- `configs/` - Configuration files (active)
- `data/` - Original data directory (may contain useful historical data)
- `outputs/` - Analysis outputs (potentially useful)
- `logs/` - Runtime logs (useful for debugging)
- `pyproject.toml`, `requirements.txt` - Python environment
- `docker-compose.yml` - Unified Docker setup
- `README.md`, `LICENSE` - Project documentation

## Summary

**Before**: 29 items in root directory (scripts, directories, configs mixed together)

**After**: 20 items in root (cleaner, focused on active code)

**Archived**: 13 old/unused files and directories

**Result**: Clearer structure, easier maintenance, all code preserved for potential restoration
