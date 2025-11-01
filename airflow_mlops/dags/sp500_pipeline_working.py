"""
S&P 500 Complete Pipeline - Docker-Native (WORKING VERSION)

This DAG uses the actual bronze data and correct argument names verified through testing.

Data Flow:
  Bronze (spx500_usd_m1_historical.ndjson) 
    → Silver (technical.csv, microstructure.csv, volatility.csv, news_sentiment.csv)
    → Gold (spx500_features.csv, sp500_trading_signals.csv, spx500_labels_30min.csv)
    → Training (XGBoost model to MLflow)

Repository: airflow_mlops/dags/sp500_pipeline_working.py
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.task_group import TaskGroup
from docker.types import Mount

# Docker configuration
DOCKER_IMAGE = 'fx-ml-pipeline-worker:latest'
NETWORK_MODE = 'fx-ml-pipeline_ml-network'
DOCKER_URL = 'unix://var/run/docker.sock'  # Docker socket

# Volume mounts
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
    'retries': 0,
    'max_active_runs': 1,
}

# Create DAG
dag = DAG(
    'sp500_pipeline_working',
    default_args=default_args,
    description='Complete working S&P 500 pipeline with verified arguments',
    schedule_interval=None,
    catchup=False,
    tags=['production', 'ml', 'sp500', 'docker', 'verified'],
)

# ============================================================================
# STAGE 1: SILVER LAYER - PARALLEL FEATURE PROCESSING
# ============================================================================

with TaskGroup("silver_processing", tooltip="Bronze to Silver transformations", dag=dag) as silver_processing:

    process_technical = DockerOperator(
        task_id='technical_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
        command=[
            '-m', 'src_clean.data_pipelines.silver.market_technical_processor',
            '--input', '/data_clean/bronze/market/spx500_usd_m1_historical.ndjson',
            '--output', '/data_clean/silver/market/technical/spx500_technical.csv'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        mem_limit='3g',
        dag=dag,
    )

    process_microstructure = DockerOperator(
        task_id='microstructure_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
        command=[
            '-m', 'src_clean.data_pipelines.silver.market_microstructure_processor',
            '--input', '/data_clean/bronze/market/spx500_usd_m1_historical.ndjson',
            '--output', '/data_clean/silver/market/microstructure/spx500_microstructure.csv'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        mem_limit='3g',
        dag=dag,
    )

    process_volatility = DockerOperator(
        task_id='volatility_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
        command=[
            '-m', 'src_clean.data_pipelines.silver.market_volatility_processor',
            '--input', '/data_clean/bronze/market/spx500_usd_m1_historical.ndjson',
            '--output', '/data_clean/silver/market/volatility/spx500_volatility.csv'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        mem_limit='3g',
        dag=dag,
    )

    process_news_sentiment = DockerOperator(
        task_id='news_sentiment',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
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

# ============================================================================
# STAGE 2: GOLD LAYER - AGGREGATION
# ============================================================================

with TaskGroup("gold_processing", tooltip="Silver to Gold aggregation", dag=dag) as gold_processing:

    build_gold_features = DockerOperator(
        task_id='build_market_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
        command=[
            '-m', 'src_clean.data_pipelines.gold.market_gold_builder',
            '--technical', '/data_clean/silver/market/technical/spx500_technical.csv',
            '--microstructure', '/data_clean/silver/market/microstructure/spx500_microstructure.csv',
            '--volatility', '/data_clean/silver/market/volatility/spx500_volatility.csv',
            '--output', '/data_clean/gold/market/features/spx500_features.csv'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        mem_limit='3g',
        dag=dag,
    )

    build_news_signals = DockerOperator(
        task_id='build_news_signals',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
        command=[
            '-m', 'src_clean.data_pipelines.gold.news_signal_builder',
            '--input', '/data_clean/silver/news/sentiment/sp500_news_sentiment.csv',
            '--output', '/data_clean/gold/news/signals/sp500_trading_signals.csv',
            '--window', '60'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        mem_limit='3g',
        dag=dag,
    )

    generate_labels = DockerOperator(
        task_id='generate_labels',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        docker_url=DOCKER_URL,
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
        mem_limit='3g',
        dag=dag,
    )

# ============================================================================
# STAGE 3: MODEL TRAINING
# ============================================================================

train_model = DockerOperator(
    task_id='train_xgboost_model',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    docker_url=DOCKER_URL,
    command=[
        '-m', 'src_clean.training.xgboost_training_pipeline_mlflow',
        '--market-features', '/data_clean/gold/market/features/spx500_features.csv',
        '--news-signals', '/data_clean/gold/news/signals/sp500_trading_signals.csv',
        '--prediction-horizon', '30',
        '--experiment-name', 'sp500_docker_pipeline'
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    environment={
        'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
    },
    mount_tmp_dir=False,
    mem_limit='4g',
    dag=dag,
)

# ============================================================================
# DAG DEPENDENCIES
# ============================================================================

# Silver layer processes in parallel
silver_processing

# Gold layer depends on silver
silver_processing >> gold_processing

# Training depends on gold
gold_processing >> train_model
