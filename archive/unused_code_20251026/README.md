# Archived Code - October 26, 2025

This directory contains code that is no longer actively used in the main pipeline.

## Archived on: 2025-10-26

## Reason for Archival:
Cleaning up codebase to focus on the clean medallion architecture (`src_clean/`).
The following files/directories have been superseded or are no longer needed.

## Contents:

### 1. Old Pipeline Scripts
- `run_sp500_pipeline.py` - Replaced by `src_clean/run_full_pipeline.py`
- `monitor_pipeline.sh` - Old pipeline monitoring (now using MLflow)

### 2. Migration & Analysis Tools
- `migrate_to_clean_structure.py` - One-time migration script (already completed)
- `analyze_data_coverage.py` - One-time data analysis script

### 3. Development/Testing
- `test-streamlit.ipynb` - Old Streamlit testing notebook

### 4. Feature Store (Optional)
- `feature_repo/` - Feast feature store configuration
  * Only used in API inference (src_clean/api/)
  * Can be kept if real-time serving is needed
  * Archived as batch inference is the current focus

### 5. Airflow MLOps (Duplicate)
- `airflow_mlops/` - Standalone Airflow setup
  * Duplicated by docker/airflow/
  * Keeping docker/ as the single source of truth

### 6. News Tools
- `news-simulator/` - Old news article simulator for testing
- `setup_news_api_keys.sh` - Old API key setup
- `setup_news_cron.sh` - Old cron job setup

## Current Active Structure:

```
fx-ml-pipeline/
├── src_clean/                    # ✅ ACTIVE - Clean medallion architecture
│   ├── data_pipelines/          # Bronze → Silver → Gold
│   ├── training/                # XGBoost + MLflow + Model Selector
│   ├── api/                     # REST API (uses Feast)
│   └── ui/                      # Streamlit dashboards
├── scripts/                      # ✅ ACTIVE - Utility scripts
├── docker/                       # ✅ ACTIVE - Unified Docker setup
├── docs/                         # ✅ ACTIVE - Documentation
└── archive/                      # 📦 Archived code
```

## Restoration:

If any of these files are needed, they can be restored from this archive.
