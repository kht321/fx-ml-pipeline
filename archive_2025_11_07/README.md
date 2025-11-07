# Archive: November 7, 2025

This directory contains code, scripts, and files that were moved out of the main project directory during codebase housekeeping on **November 7, 2025**.

## Archive Contents

### 1. Scripts (scripts/)
**8 files** - Utility and setup scripts that are no longer actively used in the production pipeline:

- `download_sp500_data.sh` - Historical S&P 500 data download script (one-time setup)
- `process_5year_data.sh` - 5-year historical data processing (completed)
- `setup_hybrid_news_scraper.sh` - News scraper setup script (one-time setup)
- `feast_materialize.py` - Manual Feast feature materialization (replaced by DAG tasks)
- `test_feast_online.py` - Feast online feature testing (one-time validation)
- `register_model_mlflow.py` - MLflow model registration (replaced by DAG tasks)
- `generate_model_visualizations.py` - Model comparison visualization (manual reporting)
- `start_2025_collection.py` - 2025 data collection (broken imports, references old `/src/` paths)

**Status:** These scripts served their purpose during development/setup but are no longer called by active DAGs or production code.

---

### 2. Root Scripts (root_scripts/)
**7 files** - Shell scripts and Python utilities from project root:

- `start_airflow.sh` - Older Airflow startup script (superseded by `start_airflow_fixed.sh`)
- `start_news_simulator.sh` - Isolated news simulator start (docker-compose used instead)
- `start_streamlit.sh` - Isolated Streamlit start (docker-compose used instead)
- `stop_all.sh` - Service shutdown script (orphaned, docker-compose used)
- `fix_port_and_start.sh` - Wrapper script for Airflow (low value wrapper)
- `monitor_optimized_run.sh` - Run optimization monitoring (low usage)
- `send_status_report.py` - Email status report generator (not imported anywhere)

**Status:** Replaced by docker-compose orchestration or more robust alternatives.

---

### 3. Notebooks (notebooks/)
**4 Jupyter notebooks** - Exploratory analysis and experimental work:

- `N01_kz_rough_work.ipynb` (358KB) - Initial exploratory data analysis
- `N02_kz_ARIMA.ipynb` (295KB) - ARIMA model testing and evaluation
- `N03_hyperparameter_testing.ipynb` (183KB) - Hyperparameter tuning experiments
- `gemini_llm_feature_test.ipynb` (133KB) - Gemini LLM feature extraction testing

**Status:** Research notebooks, not used in production. Contain valuable exploratory work.

---

### 4. DAGs (dags/)
**3 DAG files** - Old or debug versions of Airflow DAGs:

- `sp500_ml_pipeline_v4_docker_DEBUG.py` - Debug version with extra logging
- `sp500_pipeline_working.py.disabled` - Obsolete v2/v3 pipeline (disabled)
- `sp500_training_pipeline_docker.py.disabled` - Obsolete training pipeline (disabled)

**Status:** Superseded by `sp500_ml_pipeline_v4_1_docker.py` (production DAG).

---

### 5. Tests (tests/)
**Full directory** - Test scripts that are no longer actively executed:

- `validate_implementation.py` - FinBERT integration validation
- `test_finbert_gold_layer.py` - FinBERT smoke test
- Other test utilities

**Status:** One-time validation tests, not part of CI/CD.

---

### 6. Outputs (outputs/)
**Full directory** - Historical output files from old pipeline runs:

- Feature CSVs
- Model comparison outputs
- Old training results

**Status:** Historical artifacts from development phase.

---

### 7. Docs (docs/)
**4 documentation files** - Old status reports and documentation:

- `PIPELINE_STATUS.md` - Pipeline status (outdated, October)
- `pipeline_enhancement_report.html` - Old enhancement report
- `monitor_output.log` - Monitor run output log
- `Model_Selection_Deployment_Process.md` - Model selection documentation (may be duplicated in docs/)

**Status:** Replaced by current documentation or outdated.

---

## Why These Files Were Archived

All files in this archive fall into one or more categories:

1. **One-time setup scripts** - Served their purpose during initial setup
2. **Superseded by DAG tasks** - Functionality moved into Airflow DAGs
3. **Exploratory work** - Notebooks and experiments not used in production
4. **Obsolete versions** - Replaced by newer implementations
5. **Not actively referenced** - Not imported or called by production code
6. **Docker-compose replaced** - Individual start scripts replaced by orchestration

---

## What Remains in Production

The active codebase now contains only:

**Active Scripts:**
- `start_airflow_fixed.sh` - Primary Airflow startup (fixes port 5000 conflict)
- `start_all.sh` - Multi-service startup for native mode
- `test_drift_simulation.sh` - Active testing/monitoring
- `test_pipeline_e2e.sh` - End-to-end pipeline validation
- `validate_pipeline.sh` - Pipeline validation

**Active DAGs:**
- `sp500_ml_pipeline_v4_1_docker.py` - Primary production DAG
- `sp500_ml_pipeline_v4_docker.py` - Backup production DAG
- `sp500_online_inference_pipeline.py` - Real-time inference DAG

**Active Code:**
- `/src_clean/` - Source code
- `/airflow_mlops/` - DAG definitions and configs
- `/feature_repo/` - Feast feature store
- `/docker/` - Container definitions
- `/data_clean/` - Active data pipeline (17GB)

---

## Storage Information

**Archive Contents:**
- Total files: 39
- Total size: ~2MB (code) + outputs/tests directories
- Notebooks: 963KB
- Scripts: ~50KB

**Large Data Directories (NOT included in this archive):**
- `/data/` (3.6GB) - Old data directory, kept for historical reference
- `/data_clean_5year/` (12GB) - 5-year historical training data backup
- `/archive/` (existing) - Previous archive folder with old codebases

These large data directories remain in place but are not actively used by current production pipeline.

---

## Restoration Instructions

If you need to restore any archived files:

```bash
# Restore specific script
cp archive_2025_11_07/scripts/feast_materialize.py scripts/

# Restore entire category
cp -r archive_2025_11_07/notebooks/ notebooks/

# Restore all
cp -r archive_2025_11_07/* ./
```

---

## Deletion Consideration

This archive can be safely deleted if:
1. No need for historical reference
2. Notebooks have been reviewed and insights documented
3. Production pipeline runs successfully for 30+ days
4. All team members have been notified

**Recommendation:** Keep for 90 days, then delete or move to external storage.

---

## Archive Metadata

- **Archive Date:** November 7, 2025
- **Archived By:** Automated housekeeping process
- **Repository:** fx-ml-pipeline
- **Production Version:** v4.1
- **Active DAG Count:** 3
- **Purpose:** Codebase cleanup and organization

---

**End of Archive README**
