"""
S&P 500 ML Pipeline V4 - Complete End-to-End Production DAG

This DAG implements a complete ML pipeline from bronze data to deployed model:
  1. DATA VALIDATION: Validate existing bronze data
  2. SILVER LAYER: Bronze → Silver feature engineering (parallel)
  3. GOLD LAYER: Silver → Gold aggregation + label generation
  4. TRAINING: Train XGBoost regression model
  5. DEPLOYMENT: Deploy best model to production

Repository: airflow_mlops/dags/sp500_ml_pipeline_v4_1_docker.py
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.task_group import TaskGroup
from docker.types import Mount

# Docker configuration
DOCKER_IMAGE = 'fx-ml-pipeline-worker:latest'
NETWORK_MODE = 'fx-ml-pipeline_ml-network'
DOCKER_URL = 'unix://var/run/docker.sock'

# Volume mounts - shared between all tasks
# NOTE: Change this to your working directory: /path/to/working_dir/
MOUNTS = [
    Mount(source='/path/to/working_dir/fx-ml-pipeline/data_clean', target='/data_clean', type='bind'),
    Mount(source='/path/to/working_dir/fx-ml-pipeline/src_clean', target='/app/src_clean', type='bind'),
    Mount(source='/path/to/working_dir/fx-ml-pipeline/models', target='/data_clean/models', type='bind'),
]

# Path is based on MOUNTS as we are using DockerOperator
RAW_DATA = "/data_clean/raw" 
DATAMART = "/data_clean" # TODO should rename to /datamart
BRONZE_MARKET = f"{DATAMART}/bronze/market"
BRONZE_NEWS = f"{DATAMART}/bronze/news"
SILVER_MARKET = f"{DATAMART}/silver/market"
SILVER_NEWS = f"{DATAMART}/silver/news"
GOLD_MARKET = f"{DATAMART}/gold/market"
GOLD_NEWS = f"{DATAMART}/gold/news"
MODELS_BANK = f"{DATAMART}/models"

# Raw data
# NOTE To save processing time, switch to smaller data set instead of 5 years file (significant slower)
NEWS_DATA_ZIP = "historical_5year.zip"
NEWS_DATA_DIR = f'{BRONZE_NEWS}/historical_5year/' # <-- change here
# NEWS_DATA_DIR = f'{BRONZE_NEWS}/' # <-- change here
MARKET_DATA_FILE= "spx500_usd_m1_5years.ndjson" # <-- change here
# MARKET_DATA_FILE= "spx500_usd_m1_2years.ndjson" # <-- change here
MARKET_DATA_ZIP = f"{MARKET_DATA_FILE}.zip"  # 
SKIP_NEWS_SIGNAL_BUILDING = True # Set to False if you want to train from fresh; For True to work,  GOLD_NEWS/signals/spx500_trading_signals.parquet must be presented 

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 26),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

# Create DAG
DAG_ID="sp500_ml_pipeline_v4_1_docker"
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Complete end-to-end S&P 500 ML pipeline: Bronze→Silver→Gold→Training→Deployment',
    # schedule_interval='0 2 * * *',  # Daily at 2 AM UTC
    schedule_interval=None,
    catchup=False,
    tags=['production', 'ml', 'sp500', 'end-to-end'],
) as dag:

    start = DockerOperator(
        task_id='start',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        entrypoint=[],
        docker_url=DOCKER_URL,
        command=[
            'python3', '-c',
            """
print('=== Start SP500 End to End ML Pipeline ===')
            """
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        dag=dag,
    )

    # ========================================================================
    # Stage 1: Medallion Dual Data Pipeline
    # ========================================================================
    with TaskGroup("medallion_dual_data_pipeline", tooltip="Medallion Data Pipeline") as medallion_dual_data_pipeline:
        # ===================================
        # Stage 1.1: Raw data ingestion into Bronze layer
        # ===================================
        with TaskGroup("bronze_data_lake", tooltip="Raw data ingestion into Bronze layer") as bronze_data_lake:

            ingest_market = DockerOperator(
                task_id='ingest_market_raw_to_bronze',
                image=DOCKER_IMAGE,
                api_version='auto',
                auto_remove=True,
                entrypoint=[],
                docker_url=DOCKER_URL,
                command=[
                    'python3', '-m', 'src_clean.data_pipelines.bronze.ingest_raw_data',
                    '--raw-data-file', os.path.join(RAW_DATA, MARKET_DATA_ZIP),
                    '--bronze-path', BRONZE_MARKET,
                ],
                mounts=MOUNTS,
                network_mode=NETWORK_MODE,
                mount_tmp_dir=False,
                dag=dag,
            )

            ingest_news = DockerOperator(
                task_id='ingest_news_raw_to_bronze',
                image=DOCKER_IMAGE,
                api_version='auto',
                auto_remove=True,
                entrypoint=[],
                docker_url=DOCKER_URL,
                command=[
                    'python3', '-m', 'src_clean.data_pipelines.bronze.ingest_raw_data',
                    '--raw-data-file', os.path.join(RAW_DATA, NEWS_DATA_ZIP),
                    '--bronze-path', BRONZE_NEWS,
                ],
                mounts=MOUNTS,
                network_mode=NETWORK_MODE,
                mount_tmp_dir=False,
                dag=dag,
            )

            validate_bronze_data = DockerOperator(
                task_id='validate_bronze_data',
                image=DOCKER_IMAGE,
                api_version='auto',
                auto_remove=True,
                entrypoint=[],
                docker_url=DOCKER_URL,
                command=[
                    'python3', '-m', 'src_clean.data_pipelines.bronze.validate_bronze_data',
                    '--data-file', os.path.join(BRONZE_MARKET, MARKET_DATA_FILE),
                ],
                mounts=MOUNTS,
                network_mode=NETWORK_MODE,
                mount_tmp_dir=False,
                dag=dag,
            )

            [ingest_market, ingest_news] >> validate_bronze_data

        # ============================================================================
        # STAGE 1.2: Build silver-layer: data cleaning, filtering, augmentation
        # ============================================================================
        with TaskGroup("silver_data_mart", tooltip="Build silver-layer: data cleaning, filtering, augmentation") as silver_data_mart:
            # ===================================
            # STAGE 1.2.1: Market data processing
            # ===================================
            with TaskGroup("market_silver", tooltip="Process market data at silver layer") as market_silver:

                process_technical = DockerOperator(
                    task_id='process_technical',
                    image=DOCKER_IMAGE,
                    api_version='auto',
                    auto_remove=True,
                    entrypoint=[],
                    docker_url=DOCKER_URL,
                    command=[
                        'python3', '-m', 'src_clean.data_pipelines.silver.market_technical_processor',
                        '--input', os.path.join(BRONZE_MARKET, MARKET_DATA_FILE),
                        '--output', f"{SILVER_MARKET}/data_clean/silver/market/technical/spx500_technical.csv"
                    ],
                    mounts=MOUNTS,
                    network_mode=NETWORK_MODE,
                    mount_tmp_dir=False,
                    mem_limit='12g',
                    dag=dag,
                )

                process_microstructure = DockerOperator(
                    task_id='process_microstructure',
                    image=DOCKER_IMAGE,
                    api_version='auto',
                    auto_remove=True,
                    entrypoint=[],
                    docker_url=DOCKER_URL,
                    command=[
                        'python3', '-m', 'src_clean.data_pipelines.silver.market_microstructure_processor',
                        '--input', os.path.join(BRONZE_MARKET, MARKET_DATA_FILE),
                        '--output', f"{SILVER_MARKET}/microstructure/spx500_microstructure.csv"
                    ],
                    mounts=MOUNTS,
                    network_mode=NETWORK_MODE,
                    mount_tmp_dir=False,
                    mem_limit='10g',
                    dag=dag,
                )

                process_volatility = DockerOperator(
                    task_id='process_volatility',
                    image=DOCKER_IMAGE,
                    api_version='auto',
                    auto_remove=True,
                    entrypoint=[],
                    docker_url=DOCKER_URL,
                    command=[
                        'python3', '-m', 'src_clean.data_pipelines.silver.market_volatility_processor',
                        '--input', os.path.join(BRONZE_MARKET, MARKET_DATA_FILE),
                        '--output', f"{SILVER_MARKET}/volatility/spx500_volatility.csv"
                    ],
                    mounts=MOUNTS,
                    network_mode=NETWORK_MODE,
                    mount_tmp_dir=False,
                    mem_limit='10g',
                    dag=dag,
                )

                # Run market features in parallel
                [process_technical, process_microstructure, process_volatility]

            # ===================================
            # STAGE 1.2.2: News data processing
            # ===================================
            with TaskGroup("news_silver", tooltip="Process news data at silver layer") as news_silver:

                process_news_sentiment = DockerOperator(
                    task_id='process_news_sentiment',
                    image=DOCKER_IMAGE,
                    api_version='auto',
                    auto_remove=True,
                    entrypoint=[],
                    docker_url=DOCKER_URL,
                    command=[
                        'python3', '-m', 'src_clean.data_pipelines.silver.news_sentiment_processor',
                        '--input-dir', NEWS_DATA_DIR,
                        '--output', f"{SILVER_NEWS}/sentiment/sp500_news_sentiment.csv"
                    ],
                    mounts=MOUNTS,
                    network_mode=NETWORK_MODE,
                    mount_tmp_dir=False,
                    mem_limit='8g',
                    dag=dag,
                )

        # ============================================================================
        # STAGE 1.3: Gold Datamart - Features Engineering, Business-lelve Aggregation and Label Generation
        # ============================================================================
        with TaskGroup("gold_data_mart", tooltip="Features Engineering, Business-lelve Aggregation and Label Generation", dag=dag) as gold_data_mart:
            # ===================================
            # STAGE 1.3.1: Feature store
            # ===================================
            with TaskGroup(group_id="gold_feature_store") as gold_feature_store:

                build_market_features = DockerOperator(
                    task_id='build_market_features',
                    image=DOCKER_IMAGE,
                    api_version='auto',
                    auto_remove=True,
                    entrypoint=[],
                    docker_url=DOCKER_URL,
                    command=[
                        'python3', '-m', 'src_clean.data_pipelines.gold.market_gold_builder',
                        '--technical', f"{SILVER_MARKET}/technical/spx500_technical.csv",
                        '--microstructure', f"{SILVER_MARKET}/microstructure/spx500_microstructure.csv",
                        '--volatility', f"{SILVER_MARKET}/volatility/spx500_volatility.csv",
                        '--output', f"{GOLD_MARKET}/features/spx500_features.csv"
                    ],
                    mounts=MOUNTS,
                    network_mode=NETWORK_MODE,
                    mount_tmp_dir=False,
                    mem_limit='6g',
                    dag=dag,
                )

                build_news_signals = DockerOperator(
                    task_id='build_news_signals',
                    image=DOCKER_IMAGE,
                    api_version='auto',
                    auto_remove=True,
                    entrypoint=[],
                    docker_url=DOCKER_URL,
                    command=[
                        'python3', '-m', 'src_clean.data_pipelines.gold.news_signal_builder',
                        '--silver-sentiment', f"{SILVER_NEWS}/sentiment/sp500_news_sentiment.csv",
                        '--bronze-news', NEWS_DATA_DIR,
                        '--output', f"{GOLD_NEWS}/signals/spx500_trading_signals.parquet",
                        '--window', '60'
                        '--skip-training', SKIP_NEWS_SIGNAL_BUILDING
                    ],
                    mounts=MOUNTS,
                    network_mode=NETWORK_MODE,
                    mount_tmp_dir=False,
                    mem_limit='8g',
                    dag=dag,
                )

                [build_market_features, build_news_signals]

            # ===================================
            # STAGE 1.3.2: Label store
            # ===================================
            with TaskGroup(group_id="gold_label_store") as gold_label_store:

                generate_labels = DockerOperator(
                    task_id='generate_labels_30min',
                    image=DOCKER_IMAGE,
                    api_version='auto',
                    auto_remove=True,
                    entrypoint=[],
                    docker_url=DOCKER_URL,
                    command=[
                        'python3', '-m', 'src_clean.data_pipelines.gold.label_generator',
                        '--input', '/data_clean/gold/market/features/spx500_features.csv',
                        '--output', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
                        '--horizon', '30',
                        # '--threshold', '0.0'
                    ],
                    mounts=MOUNTS,
                    network_mode=NETWORK_MODE,
                    mount_tmp_dir=False,
                    mem_limit='6g',
                    dag=dag,
                )

                # ===================================
                # STAGE 1.3.3: Gold data quality validation
                # ===================================
                validate_gold_quality = DockerOperator(
                    task_id='validate_gold_data_quality',
                    image=DOCKER_IMAGE,
                    api_version='auto',
                    auto_remove=True,
                    entrypoint=[],
                    docker_url=DOCKER_URL,
                    command=[
                        'python3', '-m', 'src_clean.data_pipelines.gold.validate_gold_data_quality',
                        '--gold-market-features-file', f"{GOLD_MARKET}/features/spx500_features.csv",
                        '--gold-market-labels-file', f"{GOLD_MARKET}/labels/spx500_labels_30min.csv",
                        '--gold-news-signal-file', f"{GOLD_NEWS}/signals/sp500_trading_signals.csv",
                    ],                    
                    mounts=MOUNTS,
                    network_mode=NETWORK_MODE,
                    mount_tmp_dir=False,
                    dag=dag,
                )

                generate_labels >> validate_gold_quality

            # Gold features must be built before labels
            gold_feature_store >> gold_label_store

    # ============================================================================
    # STAGE 2: Model Training
    # ============================================================================
    with TaskGroup(group_id="model_training") as model_training:

        # Train XGBoost Model
        train_xgboost_model = DockerOperator(
            task_id='train_xgboost_regression',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            entrypoint=[],
            docker_url=DOCKER_URL,
            command=[
                'python3', '-m', 'src_clean.training.xgboost_training_pipeline_mlflow',
                '--market-features', f"{GOLD_MARKET}/features/spx500_features.csv",
                '--news-signals', f"{GOLD_NEWS}/signals/sp500_trading_signals.parquet",
                '--labels', f"{GOLD_MARKET}/labels/spx500_labels_30min.csv",
                '--prediction-horizon', '30',
                '--task', 'regression',
                '--output-dir', f"{MODELS_BANK}/xgboost",
                '--experiment-name', 'sp500_xgboost_v4',
                '--mlflow-uri', 'http://ml-mlflow:5000'
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

        # Train LightGBM Model
        train_lightgbm_model = DockerOperator(
            task_id='train_lightgbm_regression',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            entrypoint=[],
            docker_url=DOCKER_URL,
            command=[
                'python3', '-m', 'src_clean.training.lightgbm_training_pipeline_mlflow',
                '--market-features', f"{GOLD_MARKET}/features/spx500_features.csv",
                '--news-signals', f"{GOLD_NEWS}/signals/sp500_trading_signals.parquet",
                '--labels', f"{GOLD_MARKET}/labels/spx500_labels_30min.csv",
                '--prediction-horizon', '30',
                '--task', 'regression',
                '--output-dir', f"{MODELS_BANK}/lightgbm",
                '--experiment-name', 'sp500_lightgbm_v4',
                '--mlflow-uri', 'http://ml-mlflow:5000'
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

        # Train ARIMAX Model
        # train_arima_model = DockerOperator(
        #     task_id='train_arima_regression',
        #     image=DOCKER_IMAGE,
        #     api_version='auto',
        #     auto_remove=True,
        #     entrypoint=[],
        #     docker_url=DOCKER_URL,
        #     command=[
        #         'python3', '-m', 'src_clean.training.arima_training_pipeline_mlflow',
        #         '--market-features', f"{GOLD_MARKET}/features/spx500_features.csv",
        #         '--news-signals', f"{GOLD_NEWS}/signals/sp500_trading_signals.parquet",
        #         '--labels', f"{GOLD_MARKET}/labels/spx500_labels_30min.csv",
        #         '--prediction-horizon', '30',
        #         '--output-dir', f"{MODELS_BANK}/arima",
        #         '--experiment-name', 'sp500_arimax_v4',
        #         '--mlflow-uri', 'http://ml-mlflow:5000'
        #     ],
        #     mounts=MOUNTS,
        #     network_mode=NETWORK_MODE,
        #     environment={
        #         'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
        #     },
        #     mount_tmp_dir=False,
        #     mem_limit='4g',
        #     dag=dag,
        # )

        # [train_xgboost_model, train_lightgbm_model, train_arima_model]
        [train_xgboost_model, train_lightgbm_model]

    # ============================================================================
    # STAGE 3: Model Selection & Deployment
    # ============================================================================
    with TaskGroup(group_id="model_selection_deployment") as model_selection_deployment:
        # ============================================================================
        # STAGE 3.1: Model Selection
        # ============================================================================
        select_best_model = DockerOperator(
            task_id='select_best_model_by_rmse',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            entrypoint=[],
            docker_url=DOCKER_URL,
            command=[
                'python3', '-m', 'src_clean.deployment.select_best_model_by_rmse',
                '--model-path', MODELS_BANK,
            ],
            mounts=MOUNTS,
            network_mode=NETWORK_MODE,
            mount_tmp_dir=False,
            dag=dag,
        )

        # ============================================================================
        # STAGE 3.2: Model Validation & Registration
        # ============================================================================
        validate_model_output = DockerOperator(
            task_id='validate_model_output',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            entrypoint=[],
            docker_url=DOCKER_URL,
            command=[
                'python3', '-m', 'src_clean.deployment.validate_model_output',
                '--prod-path', f"{MODELS_BANK}/production",
            ],
            mounts=MOUNTS,
            network_mode=NETWORK_MODE,
            mount_tmp_dir=False,
            dag=dag,
        )

        register_model_mlflow = DockerOperator(
            task_id='register_model_to_mlflow',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            entrypoint=[],
            docker_url=DOCKER_URL,
            command=[
                'python3', '-m', 'src_clean.deployment.register_model_to_mlflow',
                '--prod-path', f"{MODELS_BANK}/production",
                '--mlflow-uri', 'http://ml-mlflow:5000'
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
        # STAGE 3.2: Model Deployment
        # ============================================================================
        deploy_model = DockerOperator(
            task_id='deploy_model_to_production',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            entrypoint=[],
            docker_url=DOCKER_URL,
            command=[
                'python3', '-m', 'src_clean.deployment.deploy_model_to_production',
                '--model-path', MODELS_BANK,
                '--prod-path', f"{MODELS_BANK}/production",
            ],
            mounts=MOUNTS,
            network_mode=NETWORK_MODE,
            mount_tmp_dir=False,
            dag=dag,
        )

        select_best_model >> validate_model_output >> register_model_mlflow >> deploy_model

    # ============================================================================
    # STAGE 4: Monitoring & Drift Detection
    # ============================================================================
    with TaskGroup(group_id="model_monitoring") as model_monitoring:
        generate_monitoring_report = DockerOperator(
            task_id='generate_evidently_report',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            entrypoint=[],
            docker_url=DOCKER_URL,
            command=[
                'python3', '-m', 'src_clean.monitoring.generate_monitoring_report',
                '--gold-market-features-file', f"{GOLD_MARKET}/features/spx500_features.csv",
            ],
            mounts=MOUNTS,
            network_mode=NETWORK_MODE,
            mount_tmp_dir=False,
            dag=dag,
        )

# ============================================================================
# DAG DEPENDENCIES
# ============================================================================

start >> [ingest_market, ingest_news]
bronze_data_lake >> silver_data_mart
market_silver >> build_market_features
news_silver >> build_news_signals
gold_data_mart >> model_training >> model_selection_deployment >> model_monitoring

