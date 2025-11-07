# Changelog

All notable changes to the S&P 500 ML Pipeline project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [4.1.0] - 2025-11-05

### Added
- **Alternative Training DAG**: `sp500_ml_pipeline_v4_1_docker.py` as failsafe pipeline variant
- **Model Selection Dashboard**: Enhanced Streamlit UI with automated model selection based on test RMSE
- **Comprehensive Documentation**: Complete MLOps technical report with system architecture details
- **Professional Repository Structure**:
  - Added `CONTRIBUTING.md` with development guidelines
  - Added `CHANGELOG.md` for version tracking
  - Created `scripts/` directory for shell scripts organization
  - Added `.env.monitoring.example` template for alerting configuration

### Changed
- **Directory Structure Cleanup**: Consolidated legacy directories into `archive_2025_11_07/`
  - Archived `data/` (3.6 GB legacy data)
  - Archived `data_clean_5year/` (12 GB backup data)
  - Archived `configs/` (legacy YAML configs)
- **Code Cleanup**: Removed unused fallback paths in `src_clean/api/inference.py`
- **Documentation Updates**: Fixed broken references in README.md
- **Standardized Paths**: Updated news data path from `data/news/gold/` to `data_clean/gold/news/`

### Fixed
- Docker build cache issues causing parent snapshot errors
- Incorrect DAG filename references in documentation
- Missing `.env.monitoring.example` template file
- Legacy code references to deprecated directory structure

### Performance
- Freed 15.6 GB of active project space through archival
- Improved Docker build times with cleaned cache
- Reduced IDE indexing time with simplified project structure

---

## [4.0.0] - 2025-11-04

### Added
- **Multi-Model Training Pipeline**: Automated training and selection across XGBoost, LightGBM, and AR models
- **2-Stage Optuna Optimization**:
  - Stage 1: 50 trials for broad hyperparameter search
  - Stage 2: 50 trials for fine-tuning around optimal parameters
- **Online Inference DAG**: Real-time prediction pipeline with 5-minute scheduling
- **Model Lifecycle Management**: 4-stage deployment (Staging → Production → Archive)
- **Automated Model Selection**: Selects best model based on test RMSE performance
- **Enhanced Feature Engineering**:
  - 76 market features (technical, microstructure, volatility)
  - 11 news sentiment features from FinBERT
- **Production Model Registry**: Dedicated directory for deployed models
- **SHAP Explainability**: Feature importance analysis for all trained models

### Changed
- **Upgraded Model Training**: From single-model to multi-model selection framework
- **MLflow Integration**: Enhanced experiment tracking with detailed hyperparameter logging
- **Feature Store**: Expanded Feast integration with 87 total features
- **Data Pipeline**: Optimized medallion architecture (Bronze → Silver → Gold)

### Performance
- **FinBERT Optimization**: 20-30x speedup through batch processing
- **Pipeline Runtime**: Reduced to 25-35 minutes for full execution
- **Inference Latency**: Maintained <100ms per prediction
- **Model Training**: Parallelized across multiple model types

---

## [3.0.0] - 2025-10-15

### Added
- **FinBERT Sentiment Analysis**: Deep learning NLP for financial news
- **News Feature Pipeline**:
  - Multi-source aggregation (NewsAPI, Financial Modeling Prep, Polygon)
  - Real-time sentiment scoring
  - Historical 5-year news backfill
- **News Simulator**: Testing tool for streaming positive/negative news
- **Evidently AI Drift Detection**: Automated model monitoring
- **Email Alerting**: SMTP-based alerts for drift and performance degradation
- **Drift Dashboard**: HTML report generation with visual drift analysis

### Changed
- Expanded data sources from market-only to market + news
- Integrated NLP features into training pipeline
- Enhanced monitoring with sentiment-based drift detection

### Performance
- Initial FinBERT processing: ~5 minutes for 1000 articles (before optimization)
- News pipeline: 15-20 minutes for daily processing

---

## [2.0.0] - 2025-09-01

### Added
- **Dockerized Infrastructure**: 16 containerized services
  - Apache Airflow (webserver, scheduler, dag-processor)
  - MLflow tracking server
  - FastAPI REST API
  - Streamlit dashboard
  - PostgreSQL, Redis, Nginx
- **Feast Feature Store**: Online feature serving
- **Blue-Green Deployment**: Zero-downtime model updates
- **Health Monitoring**: API endpoints for system status
- **Comprehensive Logging**: Centralized logs/ directory

### Changed
- Migrated from local scripts to Airflow orchestration
- Replaced manual tracking with MLflow experiment management
- Introduced microservices architecture

### Performance
- Reduced deployment downtime to zero with blue-green strategy
- Improved scalability with containerized services

---

## [1.0.0] - 2025-07-01

### Added
- **Initial Pipeline Implementation**: Basic ML pipeline for S&P 500 prediction
- **XGBoost Baseline Model**: Regression model with manual hyperparameters
- **Market Data Ingestion**: OANDA API integration
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Basic Data Structure**: Bronze/Silver/Gold layers
- **Jupyter Notebooks**: Exploratory analysis and prototyping

### Features
- 30-minute ahead price predictions
- 45+ market features
- Local model storage
- CSV-based data persistence

---

## [Unreleased]

### Planned
- **LSTM/Transformer Models**: Deep learning time series forecasting
- **Reinforcement Learning**: Adaptive trading strategy optimization
- **Real-time Streaming**: Kafka integration for live data
- **Cloud Deployment**: AWS/GCP infrastructure
- **A/B Testing Framework**: Multi-model champion/challenger evaluation
- **Advanced Drift Detection**: Statistical test enhancements

---

## Version History Summary

| Version | Release Date | Key Features |
|---------|-------------|--------------|
| **4.1.0** | 2025-11-05 | Alternative DAG, cleanup, documentation |
| **4.0.0** | 2025-11-04 | Multi-model selection, 2-stage Optuna |
| **3.0.0** | 2025-10-15 | FinBERT, news features, drift detection |
| **2.0.0** | 2025-09-01 | Docker, Airflow, MLflow, Feast |
| **1.0.0** | 2025-07-01 | Initial baseline pipeline |

---

## Breaking Changes

### v4.0.0 → v4.1.0
- **Directory Structure**: Legacy `data/`, `data_clean_5year/`, and `configs/` directories moved to `archive_2025_11_07/`
  - **Migration**: Update any external scripts referencing old paths
- **News Path**: Changed from `data/news/gold/news_signals/` to `data_clean/gold/news/signals/`
  - **Migration**: Update any hardcoded paths in custom scripts

### v3.0.0 → v4.0.0
- **Model Selection**: Manual model specification replaced with automated selection
  - **Migration**: Remove hardcoded model paths, use production registry
- **Training DAG**: Single model training replaced with multi-model pipeline
  - **Migration**: Update DAG triggers and monitoring

### v2.0.0 → v3.0.0
- **Feature Count**: Increased from 45 to 87 features
  - **Migration**: Retrain all models with new feature set
- **Data Schema**: Added news sentiment columns
  - **Migration**: Update feature engineering scripts

### v1.0.0 → v2.0.0
- **Infrastructure**: Moved from local execution to Docker containers
  - **Migration**: Install Docker and Docker Compose
- **Orchestration**: Manual scripts replaced with Airflow DAGs
  - **Migration**: Port custom workflows to Airflow task structure

---

## Deprecations

### Deprecated in 4.1.0
- `data/` directory structure (use `data_clean/` instead)
- `configs/*.yaml` files (use `config/drift_thresholds.json` and environment variables)
- Legacy model paths in `src_clean/api/inference.py`

### Deprecated in 4.0.0
- Single-model training scripts (use multi-model DAGs)
- Manual hyperparameter tuning (use Optuna optimization)

### Deprecated in 3.0.0
- Market-only feature sets (use combined market + news features)

---

## Migration Guides

### Upgrading to 4.1.0 from 4.0.0

1. **Update Environment**:
   ```bash
   cp .env.monitoring.example .env.monitoring
   # Edit .env.monitoring with your SMTP credentials
   ```

2. **Verify Paths**:
   ```bash
   # Ensure data_clean/ exists and is populated
   ls -la data_clean/bronze/market/
   ls -la data_clean/gold/news/signals/
   ```

3. **Rebuild Docker**:
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

4. **Test DAGs**:
   ```bash
   # Trigger alternative DAG
   docker-compose exec airflow-scheduler airflow dags trigger sp500_ml_pipeline_v4_1_docker
   ```

### Upgrading to 4.0.0 from 3.0.0

1. **Update Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Clear Old Models**:
   ```bash
   # Backup existing models
   mv data_clean/models data_clean/models_v3_backup
   ```

3. **Run Multi-Model Training**:
   ```bash
   docker-compose exec airflow-scheduler airflow dags trigger sp500_ml_pipeline_v4_docker
   ```

4. **Verify Model Selection**:
   - Check Streamlit dashboard at http://localhost:8501
   - Verify best model selected based on RMSE

---

## Contributors

- Kevin Taukoor ([@kht321](https://github.com/kht321))
- Project maintained as part of MLE Group

---

## License

This project is licensed under the terms specified in [LICENSE](LICENSE).

For detailed technical documentation, see [docs/Technical_Report_MLOps.md](docs/Technical_Report_MLOps.md).
