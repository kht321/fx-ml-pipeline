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

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.task_group import TaskGroup
from docker.types import Mount

# Docker configuration
DOCKER_IMAGE = 'fx-ml-pipeline-worker:latest'
NETWORK_MODE = 'fx-ml-pipeline_ml-network'
DOCKER_URL = 'unix://var/run/docker.sock'

# Locate an existing news signals artifact (CSV fallback if Parquet missing)
NEWS_SIGNAL_CANDIDATES = [
    Path('/data_clean/gold/news/signals/sp500_trading_signals.parquet'),
    Path('/data_clean/gold/news/signals/sp500_trading_signals.csv'),
    Path('/data_clean/gold/news/signals/trading_signals_new.csv'),
]


def resolve_news_signals_path() -> str:
    """Pick the first available news signal artifact."""
    for candidate in NEWS_SIGNAL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    # Default to CSV variant so training scripts still receive a usable path
    return str(NEWS_SIGNAL_CANDIDATES[1])


NEWS_SIGNALS_PATH = resolve_news_signals_path()

# Volume mounts - shared between all tasks
MOUNTS = [
    Mount(source='/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/data_clean',
          target='/data_clean', type='bind'),
    Mount(source='/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/src_clean',
          target='/app/src_clean', type='bind'),
    Mount(source='/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/models',
          target='/models', type='bind'),
]

# Default arguments
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
dag = DAG(
    'sp500_ml_pipeline_DEBUG_fast',
    default_args=default_args,
    description='[DEBUG - NO FINBERT] Fast pipeline for testing: Bronze→Silver→Gold→Training→Deployment',
    schedule_interval=None,  # Manual only
    catchup=False,
    tags=['debug', 'fast', 'no-finbert'],
)

# ============================================================================
# STAGE 1: DATA VALIDATION
# ============================================================================

validate_bronze_data = DockerOperator(
    task_id='validate_bronze_data',
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

# Check market data
market_file = Path('/data_clean/bronze/market/spx500_usd_m1_5years.ndjson')
if not market_file.exists():
    print('ERROR: Market data not found at', market_file)
    sys.exit(1)

# Load and validate
df = pd.read_json(market_file, lines=True)
if len(df) < 1000:
    print(f'WARNING: Only {len(df)} market rows found')
    sys.exit(1)

print(f'✓ Market data validated: {len(df):,} rows')
print(f'✓ Date range: {df.time.min()} to {df.time.max()}')
print(f'✓ Columns: {list(df.columns)}')
        """
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    mount_tmp_dir=False,
    dag=dag,
)

# ============================================================================
# STAGE 2: SILVER LAYER - PARALLEL FEATURE PROCESSING
# ============================================================================

with TaskGroup("silver_processing", tooltip="Bronze→Silver feature engineering", dag=dag) as silver_processing:

    process_technical = DockerOperator(
        task_id='technical_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        entrypoint=[],
        docker_url=DOCKER_URL,
        command=[
            'python3', '-m', 'src_clean.data_pipelines.silver.market_technical_processor',
            '--input', '/data_clean/bronze/market/spx500_usd_m1_5years.ndjson',
            '--output', '/data_clean/silver/market/technical/spx500_technical.csv'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        mem_limit='12g',
        dag=dag,
    )

    process_microstructure = DockerOperator(
        task_id='microstructure_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        entrypoint=[],
        docker_url=DOCKER_URL,
        command=[
            'python3', '-m', 'src_clean.data_pipelines.silver.market_microstructure_processor',
            '--input', '/data_clean/bronze/market/spx500_usd_m1_5years.ndjson',
            '--output', '/data_clean/silver/market/microstructure/spx500_microstructure.csv'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        mem_limit='10g',
        dag=dag,
    )

    process_volatility = DockerOperator(
        task_id='volatility_features',
        image=DOCKER_IMAGE,
        api_version='auto',
        auto_remove=True,
        entrypoint=[],
        docker_url=DOCKER_URL,
        command=[
            'python3', '-m', 'src_clean.data_pipelines.silver.market_volatility_processor',
            '--input', '/data_clean/bronze/market/spx500_usd_m1_5years.ndjson',
            '--output', '/data_clean/silver/market/volatility/spx500_volatility.csv'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        mem_limit='10g',
        dag=dag,
    )

    # DISABLED FOR FAST DEBUG - FinBERT takes 60+ minutes
    # process_news_sentiment = DockerOperator(
    #     task_id='news_sentiment',
    #     image=DOCKER_IMAGE,
    #     api_version='auto',
    #     auto_remove=True,
    #     entrypoint=[],
    #     docker_url=DOCKER_URL,
    #     command=[
    #         'python3', '-m', 'src_clean.data_pipelines.silver.news_sentiment_processor',
    #         '--input-dir', '/data_clean/bronze/news/historical_5year/',
    #         '--output', '/data_clean/silver/news/sentiment/sp500_news_sentiment.csv'
    #     ],
    #     mounts=MOUNTS,
    #     network_mode=NETWORK_MODE,
    #     mount_tmp_dir=False,
    #     mem_limit='8g',
    #     dag=dag,
    # )

# ============================================================================
# STAGE 3: GOLD LAYER - AGGREGATION & LABEL GENERATION
# ============================================================================

with TaskGroup("gold_processing", tooltip="Silver→Gold aggregation + labels", dag=dag) as gold_processing:

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
        mem_limit='16g',
        dag=dag,
    )

    # DISABLED FOR FAST DEBUG - FinBERT signal aggregation takes 60+ minutes
    # build_news_signals = DockerOperator(
    #     task_id='build_news_signals',
    #     image=DOCKER_IMAGE,
    #     api_version='auto',
    #     auto_remove=True,
    #     entrypoint=[],
    #     docker_url=DOCKER_URL,
    #     command=[
    #         'python3', '-m', 'src_clean.data_pipelines.gold.news_signal_builder',
    #         '--silver-sentiment', '/data_clean/silver/news/sentiment/sp500_news_sentiment.csv',
    #         '--bronze-news', '/data_clean/bronze/news/historical_5year/',
    #         '--output', '/data_clean/gold/news/signals/sp500_trading_signals.parquet',
    #         '--window', '60'
    #     ],
    #     mounts=MOUNTS,
    #     network_mode=NETWORK_MODE,
    #     mount_tmp_dir=False,
    #     mem_limit='12g',
    #     dag=dag,
    # )

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
            '--horizon', '30'
        ],
        mounts=MOUNTS,
        network_mode=NETWORK_MODE,
        mount_tmp_dir=False,
        mem_limit='8g',
        dag=dag,
    )

    # Gold features must be built before labels
    build_market_features >> generate_labels

# ============================================================================
# STAGE 3.5: GOLD DATA QUALITY VALIDATION
# ============================================================================

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

# Check news signals (support CSV or Parquet)
news_candidates = [
    Path('/data_clean/gold/news/signals/sp500_trading_signals.parquet'),
    Path('/data_clean/gold/news/signals/sp500_trading_signals.csv'),
    Path('/data_clean/gold/news/signals/trading_signals_new.csv'),
]

news_file = next((path for path in news_candidates if path.exists()), None)

if news_file:
    if news_file.suffix == '.parquet':
        df_news = pd.read_parquet(news_file)
    else:
        df_news = pd.read_csv(news_file)
    print(f'✓ News signals loaded from {news_file.name}: {len(df_news):,} rows')
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

# ============================================================================
# STAGE 4: MODEL TRAINING (3 Models in Parallel)
# ============================================================================

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
        '--market-features', '/data_clean/gold/market/features/spx500_features.csv',
        '--news-signals', NEWS_SIGNALS_PATH,
        '--labels', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
        '--prediction-horizon', '30',
        '--task', 'regression',
        '--output-dir', '/models/xgboost',
        '--experiment-name', 'sp500_xgboost_v4',
        '--mlflow-uri', 'http://ml-mlflow:5000'
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    environment={
        'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
    },
    mount_tmp_dir=False,
    mem_limit='32g',  # Sequential training - no memory competition from other models
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
        '--market-features', '/data_clean/gold/market/features/spx500_features.csv',
        '--news-signals', NEWS_SIGNALS_PATH,
        '--labels', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
        '--prediction-horizon', '30',
        '--task', 'regression',
        '--output-dir', '/models/lightgbm',
        '--experiment-name', 'sp500_lightgbm_v4',
        '--mlflow-uri', 'http://ml-mlflow:5000'
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    environment={
        'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
    },
    mount_tmp_dir=False,
    mem_limit='16g',  # Reduced after optimizing news merge (iterrows -> merge_asof)
    dag=dag,
)

# Train ARIMAX Model
train_ar_model = DockerOperator(
    task_id='train_ar_regression',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    entrypoint=[],
    docker_url=DOCKER_URL,
    command=[
        'python3', '-m', 'src_clean.training.ar_training_pipeline_mlflow',
        '--market-features', '/data_clean/gold/market/features/spx500_features.csv',
        '--news-signals', NEWS_SIGNALS_PATH,
        '--labels', '/data_clean/gold/market/labels/spx500_labels_30min.csv',
        '--prediction-horizon', '30',
        '--lag-min', '1',
        '--lag-max', '7',
        '--output-dir', '/models/ar',
        '--experiment-name', 'sp500_ar_v4',
        '--mlflow-uri', 'http://ml-mlflow:5000'
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    environment={
        'MLFLOW_TRACKING_URI': 'http://ml-mlflow:5000'
    },
    mount_tmp_dir=False,
    mem_limit='32g',  # Increased for 2.6M predictions + large CSV save
    dag=dag,
)

# ============================================================================
# STAGE 4.5: MODEL SELECTION
# ============================================================================

select_best_model = DockerOperator(
    task_id='select_best_model_by_rmse',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    entrypoint=[],
    docker_url=DOCKER_URL,
    command=[
        'python3', '-c',
        """
import json
import shutil
from pathlib import Path

print('=== Model Selection: Comparing XGBoost, LightGBM, AR ===')

models_base = Path('/models')
model_types = ['xgboost', 'lightgbm', 'ar']

# Find metrics for each model
best_model = None
best_rmse = float('inf')
best_metrics = None
best_model_path = None

for model_type in model_types:
    model_dir = models_base / model_type
    if not model_dir.exists():
        print(f'⚠ {model_type} directory not found, skipping')
        continue

    # Find latest metrics file
    metrics_files = sorted(
        model_dir.rglob('*_metrics.json'),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not metrics_files:
        print(f'⚠ No metrics found for {model_type}')
        continue

    metrics_path = metrics_files[0]
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Get test RMSE
    test_rmse = metrics.get('test_rmse', float('inf'))
    test_mae = metrics.get('test_mae', float('inf'))

    print(f'{model_type.upper()}: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}')

    # Track best model
    if test_rmse < best_rmse:
        best_rmse = test_rmse
        best_model = model_type
        best_metrics = metrics

        # Find the corresponding model artifact
        candidate_name = metrics_path.name.replace('_metrics.json', '.pkl')
        candidate_path = metrics_path.with_name(candidate_name)

        if candidate_path.exists():
            best_model_path = candidate_path
        else:
            pkl_candidates = sorted(
                metrics_path.parent.glob('*.pkl'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            best_model_path = pkl_candidates[0] if pkl_candidates else None

if best_model is None or best_model_path is None:
    print('ERROR: No models found!')
    exit(1)

print(f'\\n✅ BEST MODEL: {best_model.upper()} (RMSE={best_rmse:.4f})')

# Copy best model to production directory
production_dir = models_base / 'production'
production_dir.mkdir(exist_ok=True)

# Copy model file
prod_model_path = production_dir / f'best_model_{best_model}.pkl'
shutil.copy2(best_model_path, prod_model_path)
print(f'✓ Copied {best_model_path.name} -> {prod_model_path.name}')

# Copy features file
features_src = best_model_path.with_name(f'{best_model_path.stem}_features.json')
if features_src.exists():
    features_dst = production_dir / f'best_model_{best_model}_features.json'
    shutil.copy2(features_src, features_dst)
    print(f'✓ Copied features file')

# Save selection metadata
selection_info = {
    'selected_model': best_model,
    'test_rmse': best_rmse,
    'test_mae': best_metrics.get('test_mae', None),
    'oot_rmse': best_metrics.get('oot_rmse', None),
    'oot_mae': best_metrics.get('oot_mae', None),
    'model_path': str(prod_model_path)
}

with open(production_dir / 'selection_info.json', 'w') as f:
    json.dump(selection_info, f, indent=2)

print('✓ Model selection complete!')
        """
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    mount_tmp_dir=False,
    dag=dag,
)

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
import json
import sys

print('=== Validating Model Selection Output ===')

# Check production directory
production_dir = Path('/models/production')
if not production_dir.exists():
    print('ERROR: Production directory not found')
    sys.exit(1)

# Check for selection info
selection_file = production_dir / 'selection_info.json'
if not selection_file.exists():
    print('ERROR: selection_info.json not found')
    sys.exit(1)

with open(selection_file) as f:
    selection = json.load(f)

print(f"✓ Selected Model: {selection['selected_model'].upper()}")
print(f"✓ Test RMSE: {selection['test_rmse']:.4f}")
print(f"✓ Test MAE: {selection.get('test_mae', 'N/A')}")
print(f"✓ OOT RMSE: {selection.get('oot_rmse', 'N/A')}")

# Check model file
model_file = Path(selection['model_path'])
if not model_file.exists():
    print(f'ERROR: Model file not found: {model_file}')
    sys.exit(1)

size_mb = model_file.stat().st_size / (1024 * 1024)
print(f'✓ Model file: {model_file.name} ({size_mb:.2f} MB)')

# Check for features file
features_file = production_dir / f"best_model_{selection['selected_model']}_features.json"
if features_file.exists():
    print(f'✓ Features file found')
else:
    print('⚠ Features file not found')

print('✓ Model validation complete!')
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

print('=== MLflow Model Registration (Best Selected Model) ===')

# Load selection info
production_dir = Path('/models/production')
selection_file = production_dir / 'selection_info.json'

if not selection_file.exists():
    print('ERROR: selection_info.json not found')
    sys.exit(1)

with open(selection_file) as f:
    selection = json.load(f)

selected_model = selection['selected_model']
model_path = Path(selection['model_path'])

print(f"Selected Model Type: {selected_model.upper()}")
print(f"Test RMSE: {selection['test_rmse']:.4f}")

# Load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f'✓ Model loaded: {type(model).__name__}')

# Set MLflow tracking
mlflow.set_tracking_uri('http://ml-mlflow:5000')
print(f'✓ MLflow URI: {mlflow.get_tracking_uri()}')

# Register model
try:
    with mlflow.start_run(run_name=f'production_{selected_model}_best'):
        # Log based on model type
        if selected_model == 'xgboost':
            mlflow.xgboost.log_model(model, 'model', registered_model_name='sp500_best_model_production')
        elif selected_model == 'lightgbm':
            mlflow.lightgbm.log_model(model, 'model', registered_model_name='sp500_best_model_production')
        else:
            mlflow.sklearn.log_model(model, 'model', registered_model_name='sp500_best_model_production')

        # Log metrics from selection
        metrics_to_log = {
            'test_rmse': selection['test_rmse'],
            'test_mae': selection.get('test_mae'),
            'oot_rmse': selection.get('oot_rmse'),
            'oot_mae': selection.get('oot_mae')
        }

        for k, v in metrics_to_log.items():
            if v is not None:
                mlflow.log_metric(k, v)

        mlflow.log_param('selected_model_type', selected_model)

        print(f'✓ Model registered successfully as sp500_best_model_production')

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

# ============================================================================
# STAGE 6: MODEL DEPLOYMENT
# ============================================================================

deploy_model = DockerOperator(
    task_id='deploy_model_to_production',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    entrypoint=[],
    docker_url=DOCKER_URL,
    command=[
        'python3', '-c',
        """
from datetime import datetime
from pathlib import Path
import json
import shutil

print('=== Model Deployment ===')

models_dir = Path('/models')
production_dir = models_dir / 'production'
selection_file = production_dir / 'selection_info.json'

if not selection_file.exists():
    print('ERROR: selection_info.json not found - run model selection first')
    exit(1)

with open(selection_file) as f:
    selection = json.load(f)

best_model_path = Path(selection['model_path'])
if not best_model_path.exists():
    # If selection stored relative path, look inside production directory
    best_model_path = production_dir / Path(selection['model_path']).name

if not best_model_path.exists():
    print(f'ERROR: Selected model artifact not found: {selection["model_path"]}')
    exit(1)

production_dir.mkdir(exist_ok=True)

# Copy best model to canonical deployment name
deployed_model = production_dir / 'current_model.pkl'
shutil.copy2(best_model_path, deployed_model)
print(f'✓ Deployed model: {best_model_path.name} -> {deployed_model.name}')

# Persist metrics snapshot from selection info
metrics_output = production_dir / 'current_metrics.json'
with open(metrics_output, 'w') as f:
    json.dump(
        {
            'selected_model': selection['selected_model'],
            'test_rmse': selection.get('test_rmse'),
            'test_mae': selection.get('test_mae'),
            'oot_rmse': selection.get('oot_rmse'),
            'oot_mae': selection.get('oot_mae'),
        },
        f,
        indent=2
    )
print('✓ Recorded deployment metrics snapshot')

# Copy feature definition if present
features_src = best_model_path.with_name(f'{best_model_path.stem}_features.json')
features_dst = production_dir / 'current_features.json'
if features_src.exists():
    shutil.copy2(features_src, features_dst)
    print('✓ Deployed feature definitions')
else:
    print('⚠ Feature definition file not found - skipping copy')

# Create deployment metadata
deployment_info = {
    'model_name': best_model_path.name,
    'selected_model_type': selection['selected_model'],
    'deployed_at': datetime.utcnow().isoformat(),
    'model_size_mb': best_model_path.stat().st_size / (1024 * 1024),
    'deployment_path': str(deployed_model)
}

with open(production_dir / 'deployment_info.json', 'w') as f:
    json.dump(deployment_info, f, indent=2)

print('✓ Model deployed to production successfully!')
print(f'Deployment info: {deployment_info}')
        """
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    mount_tmp_dir=False,
    dag=dag,
)

# ============================================================================
# STAGE 7: MONITORING & DRIFT DETECTION
# ============================================================================

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
validate_bronze_data >> silver_processing

# 2. Silver layer processes in parallel
# (all tasks in silver_processing TaskGroup run in parallel)

# 3. Gold layer depends on silver completion
silver_processing >> gold_processing

# 4. Validate gold data quality before training
gold_processing >> validate_gold_quality

# 5. Train 3 models SEQUENTIALLY to reduce peak memory (88GB -> 40GB max)
# Order: XGBoost (fastest) -> LightGBM (medium) -> AR (slowest)
validate_gold_quality >> train_xgboost_model >> train_lightgbm_model >> train_ar_model

# 6. Select best model based on RMSE (after all training completes)
train_ar_model >> select_best_model

# 7. Validate selected model output
select_best_model >> validate_model_output

# 8. Register selected model to MLflow
validate_model_output >> register_model_mlflow

# 8. Deploy model to production
register_model_mlflow >> deploy_model

# 9. Generate monitoring report after deployment
deploy_model >> generate_monitoring_report

# ============================================================================
# PIPELINE SUMMARY
# ============================================================================
#
# Complete Production ML Pipeline Flow with 3-Model Selection:
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
#   4. validate_gold_data_quality
#      ↓
#   5. model_training (3 tasks in parallel - ALL with news integration):
#      - train_xgboost_regression
#      - train_lightgbm_regression
#      - train_arima_regression (ARIMAX with exogenous news variables)
#      ↓
#   6. select_best_model_by_rmse (NEW - compares all 3 models)
#      ↓
#   7. validate_model_output
#      ↓
#   8. register_model_to_mlflow (registers best model)
#      ↓
#   9. deploy_model_to_production
#      ↓
#  10. generate_evidently_report
#
# Total Tasks: 16
#   - 1 bronze validation
#   - 4 silver processing (parallel)
#   - 3 gold processing
#   - 1 gold quality validation
#   - 3 model training (parallel - XGBoost, LightGBM, ARIMAX)
#   - 1 model selection
#   - 1 model validation
#   - 1 MLflow registration
#   - 1 production deployment
#   - 1 monitoring report
#
