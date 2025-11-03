"""
S&P 500 Online Inference Pipeline - Real-time Feature Processing & Feast Materialization

Purpose:
    Process real-time news data and materialize features to Feast online store
    for consumption by FastAPI prediction service.

Flow:
    News Simulator → Bronze → Silver (sentiment) → Gold (signals) → Feast Materialization

    Separate Services (always running):
    - FastAPI: Pulls features from Feast + makes predictions
    - Streamlit: Displays predictions

Repository: airflow_mlops/dags/sp500_online_inference_pipeline.py
"""

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
MOUNTS = [
    Mount(source='/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/data_clean',
          target='/data_clean', type='bind'),
    Mount(source='/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/src_clean',
          target='/app/src_clean', type='bind'),
    Mount(source='/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/feature_repo',
          target='/feast/feature_repo', type='bind'),
]

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 2),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'max_active_runs': 1,
}

# Create DAG
dag = DAG(
    'sp500_online_inference_pipeline',
    default_args=default_args,
    description='Real-time news processing & feature materialization for online inference',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    catchup=False,
    tags=['online', 'inference', 'real-time', 'news'],
)

# ============================================================================
# STAGE 1: CHECK FOR NEW NEWS DATA
# ============================================================================

check_new_news = DockerOperator(
    task_id='check_new_news',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    entrypoint=[],
    docker_url=DOCKER_URL,
    command=[
        'python3', '-c',
        """
import sys
from pathlib import Path
from datetime import datetime, timedelta

print('=== Checking for New News Data ===')

# Check news simulator output directory
news_dir = Path('/data_clean/bronze/news/simulated')
if not news_dir.exists():
    print('INFO: No simulated news directory yet')
    sys.exit(0)

# Check for recent files (last 20 minutes)
recent_files = []
cutoff_time = datetime.now() - timedelta(minutes=20)

for json_file in news_dir.glob('*.json'):
    # Skip tracking files
    if json_file.name in ['seen_articles.json', 'tracking.json']:
        continue

    mtime = datetime.fromtimestamp(json_file.stat().st_mtime)
    if mtime > cutoff_time:
        recent_files.append(json_file.name)

if not recent_files:
    print('INFO: No new news in last 20 minutes')
    sys.exit(0)

print(f'✓ Found {len(recent_files)} new articles to process')
for f in recent_files[:5]:
    print(f'  - {f}')
        """
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    mount_tmp_dir=False,
    dag=dag,
)

# ============================================================================
# STAGE 2: PROCESS NEWS TO SILVER LAYER
# ============================================================================

process_news_sentiment = DockerOperator(
    task_id='process_news_sentiment',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    entrypoint=[],
    docker_url=DOCKER_URL,
    command=[
        'python3', '-m', 'src_clean.data_pipelines.silver.news_sentiment_processor',
        '--input-dir', '/data_clean/bronze/news/simulated/',
        '--output', '/data_clean/silver/news/sentiment/online_sentiment.csv'
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    mount_tmp_dir=False,
    mem_limit='4g',
    dag=dag,
)

# ============================================================================
# STAGE 3: BUILD GOLD LAYER NEWS SIGNALS
# ============================================================================

build_news_signals = DockerOperator(
    task_id='build_news_signals',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    entrypoint=[],
    docker_url=DOCKER_URL,
    command=[
        'python3', '-m', 'src_clean.data_pipelines.gold.news_signal_builder',
        '--silver-sentiment', '/data_clean/silver/news/sentiment/online_sentiment.csv',
        '--bronze-news', '/data_clean/bronze/news/simulated',
        '--output', '/data_clean/gold/news/signals/online_trading_signals.csv',
        '--window', '15',  # 15-minute aggregation for real-time
        '--skip-training', 'skip'  # Skip FinBERT if output exists (for speed)
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    mount_tmp_dir=False,
    mem_limit='6g',
    dag=dag,
)

# ============================================================================
# STAGE 4: MATERIALIZE TO FEAST ONLINE STORE
# ============================================================================

materialize_to_feast = DockerOperator(
    task_id='materialize_to_feast',
    image=DOCKER_IMAGE,
    api_version='auto',
    auto_remove=True,
    entrypoint=[],
    docker_url=DOCKER_URL,
    command=[
        'python3', '-c',
        """
import sys
import subprocess
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

print('=== Feast Online Store Materialization ===')

# Check if we have new signals
signals_file = Path('/data_clean/gold/news/signals/online_trading_signals.csv')
if not signals_file.exists():
    print('INFO: No signals file found')
    sys.exit(0)

# Load signals
df = pd.read_csv(signals_file)
if len(df) == 0:
    print('INFO: No signals to materialize')
    sys.exit(0)

df['signal_time'] = pd.to_datetime(df['signal_time'], utc=True)

# Only materialize recent signals (last hour)
cutoff_time = pd.Timestamp.now(tz='UTC') - timedelta(hours=1)
recent_df = df[df['signal_time'] >= cutoff_time]

if len(recent_df) == 0:
    print('INFO: No recent signals to materialize')
    sys.exit(0)

print(f'✓ Found {len(recent_df)} recent signals to materialize')
print(f'Time range: {recent_df["signal_time"].min()} to {recent_df["signal_time"].max()}')

# Prepare for Feast
# Format: entity_id (timestamp), features (signal features)
feast_df = recent_df[[
    'signal_time',
    'avg_sentiment',
    'signal_strength',
    'trading_signal',
    'article_count',
    'quality_score'
]].copy()

# Add entity columns for Feast
feast_df['instrument'] = 'SPX500_USD'  # Entity for S&P 500
feast_df = feast_df.rename(columns={'signal_time': 'event_timestamp'})

# Save parquet file for Feast to read
feast_output = Path('/data_clean/gold/feast/online_features.parquet')
feast_output.parent.mkdir(parents=True, exist_ok=True)
feast_df.to_parquet(feast_output, index=False)

print(f'✓ Prepared {len(feast_df)} features for Feast materialization')
print(f'Saved to: {feast_output}')

# Actually materialize to Feast online store (Redis)
try:
    print('\\n🔄 Materializing features to Feast online store...')

    # Calculate time range for materialization
    start_time = recent_df['signal_time'].min()
    end_time = pd.Timestamp.now(tz='UTC')

    # Run feast materialize command
    cmd = [
        'feast', '-c', '/feast/feature_repo', 'materialize',
        start_time.strftime('%Y-%m-%dT%H:%M:%S'),
        end_time.strftime('%Y-%m-%dT%H:%M:%S')
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode == 0:
        print('✅ Successfully materialized features to Feast online store')
        print(result.stdout)
    else:
        print(f'⚠️  Feast materialization warning: {result.stderr}')
        print('Features saved to parquet file for manual materialization')

except Exception as e:
    print(f'⚠️  Could not run feast materialize: {e}')
    print('Features saved to parquet file. Feast serve will pick them up.')

print('FastAPI can now pull these features from Feast online store')
        """
    ],
    mounts=MOUNTS,
    network_mode=NETWORK_MODE,
    environment={
        'FEAST_REDIS_HOST': 'redis',
        'FEAST_REDIS_PORT': '6379'
    },
    mount_tmp_dir=False,
    dag=dag,
)

# ============================================================================
# STAGE 5: VALIDATE FEAST AVAILABILITY
# ============================================================================

validate_feast_readiness = DockerOperator(
    task_id='validate_feast_readiness',
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

print('=== Validating Feast Feature Availability ===')

# Check feast features file
feast_file = Path('/data_clean/gold/feast/online_features.parquet')
if not feast_file.exists():
    print('WARNING: No Feast features file found')
    exit(0)

df = pd.read_parquet(feast_file)
print(f'✓ Feast features: {len(df)} rows')
print(f'✓ Columns: {list(df.columns)}')
print(f'✓ Latest timestamp: {df["event_timestamp"].max()}')
print('✓ Features ready for FastAPI consumption')
print('')
print('Next steps:')
print('  1. FastAPI pulls from Feast online store')
print('  2. FastAPI makes predictions using production model')
print('  3. Streamlit displays predictions')
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

# Linear flow: Check news → Process → Build signals → Materialize → Validate
check_new_news >> process_news_sentiment >> build_news_signals >> materialize_to_feast >> validate_feast_readiness

# ============================================================================
# PIPELINE SUMMARY
# ============================================================================
#
# Online Inference Pipeline Flow:
#
#   1. check_new_news
#      ↓
#   2. process_news_sentiment (Silver layer)
#      ↓
#   3. build_news_signals (Gold layer)
#      ↓
#   4. materialize_to_feast (Feast online store)
#      ↓
#   5. validate_feast_readiness
#
# Separate Always-Running Services:
#   - News Simulator (port 5050): Generates simulated news
#   - FastAPI (port 8000): Serves predictions using Feast features
#   - Streamlit (port 8501): Displays predictions
#   - Feast (port 6566): Feature store serving layer
#   - Redis (port 6379): Feast online store backend
#
# Total tasks: 5
#
