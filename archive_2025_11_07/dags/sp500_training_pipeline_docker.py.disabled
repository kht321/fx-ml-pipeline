"""
S&P 500 Training Pipeline - Docker-Native DAG

Simplified training pipeline that assumes bronze data already exists.
Uses DockerOperator for all tasks with proper containerized execution.

Data Assumptions:
- Bronze market data exists in data_clean/bronze/market/
- Bronze news data exists in data_clean/bronze/news/

For inference (separate workflow):
- Connect to OANDA for live market data
- Use news-simulator service for real-time news

Repository: fx-ml-pipeline/airflow_mlops/dags/sp500_training_pipeline_docker.py
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.task_group import TaskGroup
from docker.types import Mount

# Docker configuration
DOCKER_IMAGE = 'fx-ml-pipeline-worker:latest'
NETWORK_MODE = 'fx-ml-pipeline_ml-network'

# Volume mounts - shared between all tasks
MOUNTS = [
    Mount(source='/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/data_clean',
          target='/data_clean', type='bind'),
    Mount(source='/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/src_clean',
          target='/app/src_clean', type='bind'),
]

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 26),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'max_active_runs': 1,
}

# Create DAG
dag = DAG(
    'sp500_training_pipeline_docker',
    default_args=default_args,
    description='Docker-native S&P 500 Training Pipeline (assumes bronze data exists)',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['production', 'ml', 'sp500', 'docker', 'training'],
)

# ============================================================================
# STAGE 1: FEATURE ENGINEERING (PARALLEL PROCESSING)
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
            '--input', '/data_clean/bronze/market/spx500_historical_5year.ndjson',
            '--output-dir', '/data_clean/silver/market/technical/'
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
            '--input', '/data_clean/bronze/market/spx500_historical_5year.ndjson',
            '--output-dir', '/data_clean/silver/market/microstructure/'
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
            '--input', '/data_clean/bronze/market/spx500_historical_5year.ndjson',
            '--output-dir', '/data_clean/silver/market/volatility/'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
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
            '--technical-dir', '/data_clean/silver/market/technical/',
            '--microstructure-dir', '/data_clean/silver/market/microstructure/',
            '--volatility-dir', '/data_clean/silver/market/volatility/',
            '--output-dir', '/data_clean/gold/market/features/'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

    # Set dependencies
    [process_technical, process_microstructure, process_volatility] >> build_gold_features

# ============================================================================
# STAGE 2: NEWS PROCESSING
# ============================================================================

process_news = DockerOperator(
    task_id='process_news_sentiment',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    command=[
        '-m', 'src_clean.data_pipelines.gold.news_signal_builder',
        '--input-dir', '/data_clean/bronze/news/',
        '--output-dir', '/data_clean/gold/news/signals/',
        '--window', '60'
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    mount_tmp_dir=False,
    dag=dag,
)

# ============================================================================
# STAGE 3: LABEL GENERATION
# ============================================================================

with TaskGroup("label_generation", tooltip="Generate training labels", dag=dag) as label_generation:

    generate_labels = DockerOperator(
        task_id='generate_30min_labels',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.data_pipelines.gold.label_generator',
            '--input', '/data_clean/gold/market/features/spx500_features.csv',
            '--output', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
            '--horizon', '30',
            '--threshold', '0.0'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

# ============================================================================
# STAGE 4: MODEL TRAINING
# ============================================================================

with TaskGroup("model_training", tooltip="Train ML models", dag=dag) as model_training:

    train_xgboost = DockerOperator(
        task_id='train_xgboost_model',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        command=[
            '-m', 'src_clean.training.xgboost_training_pipeline_mlflow',
            '--market-features', '/data_clean/gold/market/features/spx500_features.csv',
            '--news-signals', '/data_clean/gold/news/signals/sp500_trading_signals.csv',
            '--prediction-horizon', '30',
            '--experiment-name', 'sp500_xgboost_docker',
            '--model-output', '/data_clean/models/'
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
# DAG DEPENDENCIES
# ============================================================================

# Feature engineering and news processing in parallel
feature_engineering
process_news

# Label generation after features
feature_engineering >> label_generation

# Training after all preprocessing
[feature_engineering, label_generation, process_news] >> model_training

# ============================================================================
# SUCCESS CRITERIA
# ============================================================================

# This DAG assumes:
# 1. Bronze data exists in data_clean/bronze/market/ and data_clean/bronze/news/
# 2. All tasks run in isolated Docker containers
# 3. Data shared via bind mounts
# 4. MLflow accessible at http://ml-mlflow:5000
