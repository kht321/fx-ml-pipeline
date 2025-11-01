"""
S&P 500 ML Pipeline V4 - Docker-Native Production DAG

This DAG refactors V3 to use DockerOperator for ALL tasks, ensuring:
- Proper containerized execution through Airflow
- Isolated task environments with full dependencies
- Shared data volumes for pipeline artifacts
- Production-ready architecture

Key Improvements over V3:
- All BashOperator tasks replaced with DockerOperator
- Uses fx-ml-pipeline-worker:latest image with full src_clean codebase
- Proper volume mounts for data sharing between tasks
- Container-native paths (no /app, no .venv assumptions)
- Each task runs in isolated container

Repository: fx-ml-pipeline/airflow_mlops/dags/sp500_ml_pipeline_v4_docker.py
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.task_group import TaskGroup
from docker.types import Mount
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 26),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

# Docker configuration for all tasks
DOCKER_IMAGE = 'fx-ml-pipeline-worker:latest'
NETWORK_MODE = 'fx-ml-pipeline_ml-network'

# Volume mounts - shared between all tasks
# TODO change to HOST_DIR
MOUNTS = [
    Mount(source='/Users/dracbook/devroot/python/cs611/fx-ml-pipeline/data_clean',
          target='/data_clean', type='bind'),
    Mount(source='/Users/dracbook/devroot/python/cs611/fx-ml-pipeline/src_clean',
          target='/app/src_clean', type='bind'),
]

# Create DAG
dag = DAG(
    'sp500_auto_ml_pipeline_docker',
    default_args=default_args,
    description='Docker-native S&P 500 ML Pipeline with full containerization',
    # schedule_interval='0 2 * * *',  # Daily at 2 AM UTC
    schedule_interval=None,
    catchup=False,
    tags=['production', 'ml', 'sp500', 'docker'],
)

# ============================================================================
# STAGE 1: DATA COLLECTION & VALIDATION
# ============================================================================

with TaskGroup("data_collection", tooltip="Collect and validate market/news data", dag=dag) as data_collection:

    # Collect market data from OANDA
    # collect_market_data = DockerOperator(
    #     task_id='collect_market_data',
    #     image=DOCKER_IMAGE,
    #     api_version='auto',
    #     auto_remove=True,
    #     command=[
    #         '-m', 'src_clean.data_pipelines.bronze.market_data_downloader',
    #         '--instrument', 'SPX500_USD',
    #         '--granularity', 'M1',
    #         '--output-dir', '/data_clean/bronze/market/',
    #         '--years', '1'
    #     ],
    #     mounts=MOUNTS,
    #     network_mode=NETWORK_MODE,
    #     mount_tmp_dir=False,
    #     dag=dag,
    # )

    # # Collect news data with improved coverage
    # collect_news_data = DockerOperator(
    #     task_id='collect_news_data',
    #     image=DOCKER_IMAGE,
    #     api_version='auto',
    #     auto_remove=True,
    #     command=[
    #         '-m', 'src_clean.data_pipelines.bronze.hybrid_news_scraper',
    #         '--output', '/data_clean/bronze/news/',
    #         '--max-articles', '2000',
    #         '--sources', 'reuters', 'bloomberg', 'cnbc', 'wsj'
    #     ],
    #     mounts=MOUNTS,
    #     network_mode=NETWORK_MODE,
    #     mount_tmp_dir=False,
    #     dag=dag,
    # )

    # Validate data quality
    validate_data = DockerOperator(
        task_id='validate_data_quality',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-c',
            """
import pandas as pd
import sys
from pathlib import Path

# Check market data
market_file = Path('/data_clean/bronze/market/spx500_usd_m1_2years.ndjson')
if not market_file.exists():
    print('ERROR: Market data not found')
    sys.exit(1)

# Check minimum rows
df = pd.read_json(market_file, lines=True)
if len(df) < 1000:
    print(f'WARNING: Only {len(df)} market rows found')

print(f'Market data validated: {len(df)} rows')
print(f'Date range: {df.time.min()} to {df.time.max()}')
            """
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

    validate_data
    # [collect_market_data, collect_news_data] >> validate_data

# ============================================================================
# STAGE 2: FEATURE ENGINEERING (PARALLEL PROCESSING)
# ============================================================================

with TaskGroup("feature_engineering", tooltip="Process all feature types in parallel", dag=dag) as feature_engineering:

    # Silver layer - parallel processing
    process_technical = DockerOperator(
        task_id='process_technical_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.data_pipelines.silver.market_technical_processor',
            '--input', '/data_clean/bronze/market/spx500_usd_m1_2years.ndjson',
            '--output', '/data_clean/silver/market/technical/'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

    process_microstructure = DockerOperator(
        task_id='process_microstructure_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.data_pipelines.silver.market_microstructure_processor',
            '--input', '/data_clean/bronze/market/spx500_usd_m1_2years.ndjson',
            '--output', '/data_clean/silver/market/microstructure/'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

    process_volatility = DockerOperator(
        task_id='process_volatility_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.data_pipelines.silver.market_volatility_processor',
            '--input', '/data_clean/bronze/market/spx500_usd_m1_2years.ndjson',
            '--output', '/data_clean/silver/market/volatility/'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

    process_news_sentiment = DockerOperator(
            task_id='process_news_sentiment',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            # docker_url=DOCKER_URL,
            command=[
                '-m', 'src_clean.data_pipelines.silver.news_sentiment_processor',
                '--input-dir', '/data_clean/bronze/news/',
                '--output', '/data_clean/silver/news/sentiment/sp500_news_sentiment.csv'
            ],
            mounts=MOUNTS,
            network_mode=NETWORK_MODE,
            mount_tmp_dir=False,
            mem_limit='3g',
            dag=dag,
        )


    # Gold layer - combine all features
    build_gold_features = DockerOperator(
        task_id='build_gold_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.data_pipelines.gold.market_gold_builder',
            '--technical', '/data_clean/silver/market/technical/',
            '--microstructure', '/data_clean/silver/market/microstructure/',
            '--volatility', '/data_clean/silver/market/volatility/',
            '--output', '/data_clean/gold/market/features/spx500_features.csv'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )


    # Enhanced feature engineering (NEW - 114 features total!)
    enhance_features = DockerOperator(
        task_id='enhance_features_advanced',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.features.advanced_feature_engineering',
            '--input', '/data_clean/gold/market/features/spx500_features.csv',
            '--output', '/data_clean/gold/market/features/spx500_features_enhanced.csv'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

    # Set dependencies within task group
    [process_technical, process_microstructure, process_volatility] >> build_gold_features >> enhance_features


# ============================================================================
# STAGE 3: NEWS PROCESSING WITH FINBERT (OPTIMIZED MERGE)
# ============================================================================

process_news_finbert = DockerOperator(
    task_id='process_news_finbert_optimized',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    # command=[
    #     '-m', 'src_clean.data_pipelines.gold.news_signal_builder_optimized',
    #     '--input', '/data_clean/bronze/news/',
    #     '--output', '/data_clean/gold/news/signals/',
    #     '--model', 'finbert',
    #     '--window', '60',
    #     '--batch-size', '32'
    # ],    
    command=[
        '-m', 'src_clean.data_pipelines.gold.news_signal_builder',
        '--bronze-news', '/data_clean/bronze/news/',
        '--silver-sentiment', '/data_clean/silver/news/sentiment/sp500_news_sentiment.csv',
        '--output', '/data_clean/gold/news/signals/spx500_news_signals.csv',
        '--window', '60',
    ],   
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    mount_tmp_dir=False,
    dag=dag,
)

process_news_sentiment >> process_news_finbert


# ============================================================================
# STAGE 4: LABEL GENERATION (MULTIPLE HORIZONS)
# ============================================================================

with TaskGroup("label_generation", tooltip="Generate labels for multiple horizons", dag=dag) as label_generation:

    generate_30min_labels = DockerOperator(
        task_id='generate_30min_labels',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.data_pipelines.gold.label_generator',
            '--input', '/data_clean/gold/market/features/spx500_features_enhanced.csv',
            '--output', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
            '--horizon', '30',
            # '--threshold', '0.0'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

    generate_60min_labels = DockerOperator(
        task_id='generate_60min_labels',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.data_pipelines.gold.label_generator',
            '--input', '/data_clean/gold/market/features/spx500_features_enhanced.csv',
            '--output', '/data_clean/gold/market/labels/spx500_labels_60min.csv',
            '--horizon', '60',
            # '--threshold', '0.0'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

# ============================================================================
# STAGE 5: MODEL TRAINING (COMPREHENSIVE)
# ============================================================================

with TaskGroup("model_training", tooltip="Train all model variants", dag=dag) as model_training:

    # XGBoost with enhanced features (114 features)
    train_xgboost_enhanced = DockerOperator(
        task_id='train_xgboost_enhanced',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.training.xgboost_training_pipeline_mlflow',
            '--market-features', '/data_clean/gold/market/features/spx500_features_enhanced.csv',
            '--news-signals', '/data_clean/gold/news/signals/spx500_news_signals.csv',
            '--labels', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
            '--task', 'classification',
            '--output-dir', '/data_clean/models',
            '--experiment-name', 'sp500_xgboost_enhanced',
            # '--use-optimized-config'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        environment={
            'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
        },
        mount_tmp_dir=False,
        dag=dag,
    )

    # LightGBM with original features
    train_lightgbm_original = DockerOperator(
        task_id='train_lightgbm_original',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.training.lightgbm_training_pipeline_mlflow',
            '--market-features', '/data_clean/gold/market/features/spx500_features.csv',
            '--news-signals', '/data_clean/gold/news/signals/spx500_news_signals.csv',
            '--labels', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
            '--task', 'classification',
            '--output-dir', '/data_clean/models',
            '--experiment-name', 'sp500_lightgbm_original'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        environment={
            'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
        },
        mount_tmp_dir=False,
        dag=dag,
    )

# ============================================================================
# STAGE 6: MODEL SELECTION & DEPLOYMENT
# ============================================================================

select_best_model = DockerOperator(
    task_id='select_best_model',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    command=[
        '-m', 'src_clean.training.multi_experiment_selector',
        '--experiments', 'sp500_xgboost_enhanced', 'sp500_lightgbm_original',
        '--metric', 'oot_auc',
        '--min-threshold', '0.50',
        '--max-overfitting', '0.25',
        # '--output', '/data_clean/models/best_model_selection.json'
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    environment={
        'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
    },
    mount_tmp_dir=False,
    dag=dag,
)

deploy_best_model = DockerOperator(
    task_id='deploy_best_model',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    command=[
        '-c',
        """
import json
import shutil
from pathlib import Path
from datetime import datetime

# Load best model selection
with open('/data_clean/models/best_model_selection.json', 'r') as f:
    selection = json.load(f)

best_model = selection['best_model']
print(f'Deploying best model: {best_model["experiment"]} with OOT AUC: {best_model["oot_auc"]:.4f}')

# Copy model to production
src = Path(f'/data_clean/models/{best_model["model_file"]}')
dst = Path('/data_clean/models/production/current_model.pkl')
dst.parent.mkdir(exist_ok=True, parents=True)
shutil.copy2(src, dst)

# Save metadata
metadata = {
    'deployed_at': str(datetime.now()),
    'model': best_model,
    'performance': {
        'oot_auc': best_model['oot_auc'],
        'test_auc': best_model['test_auc'],
        'overfitting_ratio': best_model['overfitting_ratio']
    }
}

with open('/data_clean/models/production/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('Model deployed successfully!')
        """
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    mount_tmp_dir=False,
    dag=dag,
)

# ============================================================================
# DAG DEPENDENCIES
# ============================================================================

# Stage 1 & 2: Data collection → Feature engineering
data_collection >> feature_engineering

# Stage 3: News processing (parallel with feature engineering)
# data_collection >> process_news_finbert

# Stage 4: Label generation (needs enhanced features)
feature_engineering >> label_generation

# Stage 5: Model training (needs all preprocessing)
# NOTE: add process_news_finbert
[feature_engineering, label_generation, process_news_finbert] >> model_training

# Stage 6: Model selection and deployment
model_training >> select_best_model >> deploy_best_model

# ============================================================================
# SUCCESS CRITERIA
# ============================================================================

# Success criteria for Docker-native pipeline:
# 1. All tasks run in isolated containers
# 2. Data shared via bind mounts (not copied)
# 3. Tasks can access MLflow at http://ml-mlflow:5000
# 4. No local filesystem assumptions (/app, .venv, etc.)
# 5. Proper network connectivity via ml-network
