"""
S&P 500 ML Pipeline V4 - Complete End-to-End Production DAG

This DAG implements a complete ML pipeline from bronze data to deployed model:
  1. DATA VALIDATION: Validate existing bronze data
  2. SILVER LAYER: Bronze → Silver feature engineering (parallel)
  3. GOLD LAYER: Silver → Gold aggregation + label generation
  4. TRAINING: Train XGBoost regression model
  5. DEPLOYMENT: Deploy best model to production

Repository: airflow_mlops/dags/sp500_ml_pipeline_v4_docker.py
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from docker.types import Mount

# Docker configuration
DOCKER_IMAGE = 'fx-ml-pipeline-worker:latest'
NETWORK_MODE = 'fx-ml-pipeline_ml-network'
DOCKER_URL = 'unix://var/run/docker.sock'

# Volume mounts - shared between all tasks
# project_root = os.getcwd() # Get the current working directory
# data_clean_path = os.path.join(project_root, 'data_clean')
# absolute_path = os.path.abspath(data_clean_path) # Make sure it's an absolute path 
# absolute_path does not work for Mount!
# NOTE: Change this to your working directory: /path/to/working_dir/
MOUNTS = [
    Mount(source='/path/to/working_dir/fx-ml-pipeline/data_clean', target='/data_clean', type='bind'),
    Mount(source='/path/to/working_dir/fx-ml-pipeline/src_clean', target='/app/src_clean', type='bind'),
    Mount(source='/path/to/working_dir/fx-ml-pipeline/models', target='/models', type='bind'),
]

BASE = "/opt/airflow"
SRC = f"{BASE}/src_clean"
DATA = f"{BASE}/data_clean"
RAW_DATA = f"{DATA}/raw"
DATAMART = f"{DATA}" # keep it same as DATA for now; TODO make it a separate folder , DATA is supposed to be for raw data 
BRONZE_MARKET = f"{DATAMART}/bronze/market"
BRONZE_NEWS = f"{DATAMART}/bronze/news"
SILVER_MARKET = f"{DATAMART}/silver/market"
SILVER_NEWS = f"{DATAMART}/silver/news"
GOLD_MARKET = f"{DATAMART}/gold/market"
GOLD_NEWS = f"{DATAMART}/gold/news"
MODELS = f"{DATAMART}/models"

# raw data files and checks
NEWS_DATA_ZIP = "historical_5year.zip"
# NEWS_DATA_DIR = '/data_clean/bronze/news/historical_5year/' 
NEWS_DATA_DIR = '/data_clean/bronze/news/'

# NOTE To save processing time, switch to 2 years data instead of 5 years file (significant slower)
# MARKET_DATA_ZIP = "spx500_usd_m1_5years.ndjson.zip"
MARKET_DATA_ZIP = "spx500_usd_m1_2years.ndjson.zip" 
MARKET_DATA_FILE= "spx500_usd_m1_2years.ndjson" # <-- change here

def check_data_availability():
    """Verify that required input data exists."""
    market_zip = f"{RAW_DATA}/{MARKET_DATA_ZIP}"
    news_zip = f"{RAW_DATA}/{NEWS_DATA_ZIP}"

    if not os.path.exists(market_zip):
        raise FileNotFoundError(f"Market data not found: {market_zip}")

    if not os.path.exists(news_zip):
        raise FileNotFoundError(f"News data not found: {news_zip}")

    print(f"[check_data] Market data: {market_zip} OK")
    print(f"[check_data] News data: {news_zip} OK")
def unzip_file(src_file_path: str, dest_dir_path: str) -> None:
    """
    Unzips a single .zip file from src_file_path into dest_dir_path.
    """
    print(f"Source file: {src_file_path}")
    print(f"Destination directory: {dest_dir_path}")
    src_file = Path(src_file_path)
    dest_dir = Path(dest_dir_path)
    if not src_file.is_file():
        raise ValueError(f"[error] Source file not found: {src_file}")
    if src_file.suffix.lower() != ".zip":
        raise ValueError(f"[error] Source is not a .zip file: {src_file}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(src_file, "r") as zf:
            zf.extractall(dest_dir)
        print(f"[unzip] Successfully extracted {src_file} -> {dest_dir}")
    
    except zipfile.BadZipFile as e:
        print(f"[error] Failed to unzip {src_file}. File corrupt or invalid.")
        raise e  

def ingest_market_data() -> None:
    unzip_file(os.path.join(RAW_DATA, MARKET_DATA_ZIP), BRONZE_MARKET)
def ingest_news_data() -> None:
    unzip_file(os.path.join(RAW_DATA, NEWS_DATA_ZIP), BRONZE_NEWS)

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

DAG_ID="sp500_ml_pipeline_v4_docker"
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Complete end-to-end S&P 500 ML pipeline: Bronze→Silver→Gold→Training→Deployment',
    # schedule_interval='0 2 * * *',  # Daily at 2 AM UTC
    schedule_interval=None,
    catchup=False,
    tags=['production', 'ml', 'sp500', 'end-to-end'],
) as dag:

    start = DummyOperator(task_id="start")

    # ========================================================================
    # Stage 1: Medallion Dual Data Pipeline
    # ========================================================================
    with TaskGroup("medallion_dual_data_pipeline", tooltip="Medallion Data Pipeline") as medallion_dual_data_pipeline:
        # ===================================
        # Stage 1.1: Raw data ingestion into Bronze layer
        # ===================================
        with TaskGroup("bronze_data_lake", tooltip="Raw data ingestion into Bronze layer") as bronze_data_lake:

            check_data_availability = PythonOperator(
                task_id="check_data_availability",
                python_callable=check_data_availability,
            )

            ingest_market = PythonOperator(
                task_id="ingest_market_raw_to_bronze",
                python_callable=ingest_market_data,
            )

            ingest_news = PythonOperator(
                task_id="ingest_news_raw_to_bronze",
                python_callable=ingest_news_data,
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
                    '--data-file', os.path.join('/data_clean/bronze/market', MARKET_DATA_FILE),
                ],
                mounts=MOUNTS,
                network_mode=NETWORK_MODE,
                mount_tmp_dir=False,
                dag=dag,
            )

            check_data_availability >> [ingest_market, ingest_news] >> validate_bronze_data

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
                        '--input', os.path.join('/data_clean/bronze/market', MARKET_DATA_FILE),
                        '--output', '/data_clean/silver/market/technical/spx500_technical.csv'
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
                        '--input', os.path.join('/data_clean/bronze/market', MARKET_DATA_FILE),
                        '--output', '/data_clean/silver/market/microstructure/spx500_microstructure.csv'
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
                        '--input', os.path.join('/data_clean/bronze/market', MARKET_DATA_FILE),
                        '--output', '/data_clean/silver/market/volatility/spx500_volatility.csv'
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
                        '--output', '/data_clean/silver/news/sentiment/sp500_news_sentiment.csv'
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
                        '--technical', '/data_clean/silver/market/technical/spx500_technical.csv',
                        '--microstructure', '/data_clean/silver/market/microstructure/spx500_microstructure.csv',
                        '--volatility', '/data_clean/silver/market/volatility/spx500_volatility.csv',
                        '--output', '/data_clean/gold/market/features/spx500_features.csv'
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
                        '--silver-sentiment', '/data_clean/silver/news/sentiment/sp500_news_sentiment.csv',
                        '--bronze-news', NEWS_DATA_DIR,
                        '--output', '/data_clean/gold/news/signals/spx500_trading_signals.csv',
                        '--window', '60'
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
                        'python3', '-c',
                        """
import pandas as pd
import sys
from pathlib import Path

print('=== Gold Data Quality Validation ===')

# Check market features
features_file = Path('/data_clean/gold/market/features/spx500_features.csv')
if not features_file.exists():
    print(f'ERROR: Features file not found: {features_file}')
    sys.exit(1)

df_features = pd.read_csv(features_file)
print(f'✓ Features file loaded: {len(df_features):,} rows')

# Check for missing values
missing_pct = (df_features.isnull().sum() / len(df_features) * 100)
critical_missing = missing_pct[missing_pct > 50]
if len(critical_missing) > 0:
    print(f'ERROR: Critical missing values (>50%):')
    print(critical_missing)
    sys.exit(1)

print(f'✓ Missing values check passed (max: {missing_pct.max():.2f}%)')

# Check labels
labels_file = Path('/data_clean/gold/market/labels/spx500_labels_30min.csv')
if not labels_file.exists():
    print(f'ERROR: Labels file not found: {labels_file}')
    sys.exit(1)

df_labels = pd.read_csv(labels_file)
print(f'✓ Labels file loaded: {len(df_labels):,} rows')

# Check news signals
news_file = Path('/data_clean/gold/news/signals/sp500_trading_signals.csv')
if news_file.exists():
    df_news = pd.read_csv(news_file)
    print(f'✓ News signals loaded: {len(df_news):,} rows')
else:
    print('⚠ News signals file not found (optional)')

print('✓ Gold data quality validation PASSED')
                        """
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
    # STAGE 4: MODEL TRAINING
    # ============================================================================
    with TaskGroup(group_id="model_training") as model_training:

        train_xgboost_model = DockerOperator(
            task_id='train_xgboost_regression',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            entrypoint=[],
            docker_url=DOCKER_URL,
            command=[
                'python3', '-m', 'src_clean.training.xgboost_training_pipeline_mlflow',
                '--market-features', '/data_clean/gold/market/features/spx500_features.csv',
                '--news-signals', '/data_clean/gold/news/signals/sp500_trading_signals.csv',
                '--labels', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
                '--prediction-horizon', '30',
                '--task', 'regression',
                '--output-dir', '/models',
                '--experiment-name', 'sp500_ml_pipeline_v4_production',
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

        # LightGBM with original features
        train_lightgbm_original = DockerOperator(
            task_id='train_lightgbm_original',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            command=[
                '-m', 'src_clean.training.lightgbm_training_pipeline_mlflow',
                '--market-features', '/data_clean/gold/market/features/spx500_features.csv',
                '--news-signals', '/data_clean/gold/news/signals/sp500_trading_signals.csv',
                '--labels', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
                '--task', 'regression',
                '--output-dir', '/data_clean/models',
                '--experiment-name', 'sp500_ml_pipeline_v4_production_lightgbm'
            ],
            mounts=MOUNTS,
            network_mode=NETWORK_MODE,
            environment={
                'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
            },
            mount_tmp_dir=False,
            dag=dag,
        )

        [train_xgboost_model, train_lightgbm_original]

        # ============================================================================
        # STAGE 5: MODEL VALIDATION & REGISTRATION
        # ============================================================================

        validate_model_output = DockerOperator(
            task_id='validate_model_output',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            entrypoint=[],
            docker_url=DOCKER_URL,
            command=[
                'python3', '-c',
                """
from pathlib import Path
import sys

models_dir = Path('/models')
pkl_files = list(models_dir.glob('*.pkl'))

if not pkl_files:
    print('ERROR: No model files found in /models')
    sys.exit(1)

print(f'✓ Found {len(pkl_files)} model file(s):')
for model_file in pkl_files:
    size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f'  - {model_file.name} ({size_mb:.2f} MB)')

# Check for feature file
feature_files = list(models_dir.glob('*_features.json'))
print(f'✓ Found {len(feature_files)} feature definition file(s)')

# Check for metrics file
metrics_files = list(models_dir.glob('*_metrics.json'))
print(f'✓ Found {len(metrics_files)} metrics file(s)')

print('✓ Model training successful!')
                """
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
                'python3', '-c',
                """
import mlflow
import pickle
import json
from pathlib import Path
import sys

print('=== MLflow Model Registration ===')

# Find latest model files
models_dir = Path('/models')
pkl_files = sorted(models_dir.glob('xgboost_regression_*.pkl'), key=lambda x: x.stat().st_mtime, reverse=True)

if not pkl_files:
    print('ERROR: No model files found')
    sys.exit(1)

model_path = pkl_files[0]
model_base = model_path.stem

metrics_path = models_dir / f'{model_base}_metrics.json'
features_path = models_dir / f'{model_base}_features.json'

print(f'Model: {model_path.name}')
print(f'Metrics: {metrics_path.name if metrics_path.exists() else "NOT FOUND"}')
print(f'Features: {features_path.name if features_path.exists() else "NOT FOUND"}')

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f'✓ Model loaded: {type(model).__name__}')

# Set MLflow tracking
mlflow.set_tracking_uri('http://ml-mlflow:5000')
print(f'✓ MLflow URI: {mlflow.get_tracking_uri()}')

# Register model (simplified - just log it)
try:
    with mlflow.start_run(run_name=f'production_{model_base}'):
        mlflow.sklearn.log_model(model, 'model', registered_model_name='sp500_xgboost_production')

        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            mlflow.log_metrics(metrics)
            print(f'✓ Logged metrics: {list(metrics.keys())}')

        if features_path.exists():
            with open(features_path) as f:
                features = json.load(f)
            mlflow.log_param('n_features', len(features.get('features', [])))
            print(f'✓ Logged {len(features.get("features", []))} features')

    print('✓ Model registered to MLflow successfully!')
except Exception as e:
    print(f'⚠ MLflow registration warning: {e}')
    print('✓ Continuing despite registration issue')
                """
            ],
            mounts=MOUNTS,
            network_mode=NETWORK_MODE,
            environment={
                'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
            },
            mount_tmp_dir=False,
            dag=dag,
        )

        [train_xgboost_model, train_lightgbm_original] >> validate_model_output >> register_model_mlflow

    # ============================================================================
    # STAGE 4: MODEL TRAINING
    # ============================================================================
    with TaskGroup(group_id="model_selection_deployment") as model_selection_deployment:

        # ============================================================================
        # STAGE 6: MODEL DEPLOYMENT
        # ============================================================================

        select_best_model = DockerOperator(
            task_id='select_best_model',
            image=DOCKER_IMAGE,
            api_version='auto',
            auto_remove=True,
            command=[
                '-m', 'src_clean.training.multi_experiment_selector',
                '--experiments', 'sp500_ml_pipeline_v4_production', 'sp500_ml_pipeline_v4_production_lightgbm',
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

# NOTE: There are two deployment scripts: task_id=deployment_model vs task_id=deploy_best_model,  need to check which is the right none
#         deploy_model = DockerOperator(
#             task_id='deploy_model_to_production',
#             image=DOCKER_IMAGE,
#             api_version='auto',
#             auto_remove=True,
#             entrypoint=[],
#             docker_url=DOCKER_URL,
#             command=[
#                 'python3', '-c',
#                 """
# from pathlib import Path
# import shutil
# import json
# from datetime import datetime

# print('=== Model Deployment ===')

# models_dir = Path('/data_clean/models')
# pkl_files = sorted(models_dir.glob('xgboost_regression_*.pkl'), key=lambda x: x.stat().st_mtime, reverse=True)

# if not pkl_files:
#     print('ERROR: No models to deploy')
#     exit(1)

# latest_model = pkl_files[0]
# model_base = latest_model.stem

# # Create production directory
# prod_dir = models_dir / 'production'
# prod_dir.mkdir(exist_ok=True)

# # Copy model files to production
# prod_model = prod_dir / 'current_model.pkl'
# shutil.copy(latest_model, prod_model)
# print(f'✓ Deployed model: {latest_model.name} -> production/current_model.pkl')

# # Copy metrics if exists
# metrics_file = models_dir / f'{model_base}_metrics.json'
# if metrics_file.exists():
#     shutil.copy(metrics_file, prod_dir / 'current_metrics.json')
#     print('✓ Deployed metrics')

# # Copy features if exists
# features_file = models_dir / f'{model_base}_features.json'
# if features_file.exists():
#     shutil.copy(features_file, prod_dir / 'current_features.json')
#     print('✓ Deployed feature definitions')

# # Create deployment metadata
# deployment_info = {
#     'model_name': latest_model.name,
#     'deployed_at': datetime.utcnow().isoformat(),
#     'model_size_mb': latest_model.stat().st_size / (1024 * 1024),
#     'deployment_path': str(prod_model)
# }

# with open(prod_dir / 'deployment_info.json', 'w') as f:
#     json.dump(deployment_info, f, indent=2)

# print('✓ Model deployed to production successfully!')
# print(f'Deployment info: {deployment_info}')
#                 """
#             ],
#             mounts=MOUNTS,
#             network_mode=NETWORK_MODE,
#             mount_tmp_dir=False,
#             dag=dag,
#         )

        select_best_model >> deploy_best_model

# ============================================================================
# STAGE 7: MONITORING & DRIFT DETECTION
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
                'python3', '-c',
                """
from pathlib import Path
import pandas as pd

print('=== Evidently Monitoring Report ===')

# Check if required files exist
features_file = Path('/data_clean/gold/market/features/spx500_features.csv')
if not features_file.exists():
    print('⚠ Features file not found, skipping monitoring report')
    print('  (This is expected on first run)')
    exit(0)

df = pd.read_csv(features_file)
print(f'✓ Loaded {len(df):,} rows for monitoring')

# Simple data profile
print(f'✓ Features: {len(df.columns)} columns')
print(f'✓ Date range: {df.time.min() if "time" in df.columns else "N/A"} to {df.time.max() if "time" in df.columns else "N/A"}')
print(f'✓ Missing values: {df.isnull().sum().sum()} total')

print('✓ Monitoring data validated')
print('  Note: Full Evidently report generation available via separate service')
print('  Access at: http://localhost:8050')
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

# 1. Validate bronze data first
start >> check_data_availability

bronze_data_lake >> silver_data_mart

# 2. Silver layer processes in parallel
market_silver >> build_market_features
news_silver >> build_news_signals

# 4. Validate gold data quality before training
gold_data_mart >> model_training >> model_selection_deployment >> model_monitoring


# # 7. Register model to MLflow
# validate_model_output >> register_model_mlflow

# # 8. Deploy model to production
# register_model_mlflow >> deploy_model

# # 9. Generate monitoring report after deployment
# deploy_model >> generate_monitoring_report

# ============================================================================
# PIPELINE SUMMARY
# ============================================================================
#
# Complete Production ML Pipeline Flow:
#
#   1. validate_bronze_data
#      ↓
#   2. silver_processing (4 tasks in parallel):
#      - technical_features
#      - microstructure_features
#      - volatility_features
#      - news_sentiment
#      ↓
#   3. gold_processing (3 tasks):
#      - build_market_features
#      - build_news_signals
#      - generate_labels_30min (depends on build_market_features)
#      ↓
#   4. validate_gold_data_quality (NEW)
#      ↓
#   5. train_xgboost_regression
#      ↓
#   6. validate_model_output
#      ↓
#   7. register_model_to_mlflow (NEW)
#      ↓
#   8. deploy_model_to_production (NEW)
#      ↓
#   9. generate_evidently_report (NEW)
#
# Total Tasks: 14
#   - 1 bronze validation
#   - 4 silver processing (parallel)
#   - 3 gold processing
#   - 1 gold quality validation
#   - 1 model training
#   - 1 model validation
#   - 1 MLflow registration
#   - 1 production deployment
#   - 1 monitoring report
#