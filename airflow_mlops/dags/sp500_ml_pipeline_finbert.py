"""
S&P 500 ML Pipeline with FinBERT - Airflow DAG

Repository Location: fx-ml-pipeline/airflow_mlops/dags/sp500_ml_pipeline_finbert.py

Purpose:
    Production Airflow DAG for the complete S&P 500 ML pipeline including
    FinBERT-powered news sentiment analysis.

Pipeline Stages:
    1. Bronze → Silver (Market): Technical, microstructure, volatility features
    2. Bronze → Silver (News): TextBlob sentiment preprocessing
    3. Silver → Gold (Market): Merge market features
    4. Silver → Gold (News): FinBERT trading signals ← NEW!
    5. Gold: Generate prediction labels
    6. Train: XGBoost model with news features

Schedule:
    - Daily at 2 AM UTC (processes previous day's data)
    - Manual trigger available

Dependencies:
    - OANDA market data in bronze layer
    - News articles in bronze layer
    - FinBERT model pre-downloaded in Docker image

Usage:
    # Trigger via Airflow UI or CLI
    airflow dags trigger sp500_ml_pipeline_finbert

    # With config
    airflow dags trigger sp500_ml_pipeline_finbert \
        --conf '{"prediction_horizon": 30, "window": 60}'
"""

import os
from datetime import timedelta, datetime
import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

# Paths inside Airflow container (mapped via docker-compose)
BASE = "/opt/airflow"
SRC = f"{BASE}/src_clean"
BRONZE_MARKET = f"{BASE}/data/bronze/market"
BRONZE_NEWS = f"{BASE}/data/bronze/news"
SILVER_MARKET = f"{BASE}/data/silver/market"
SILVER_NEWS = f"{BASE}/data/silver/news"
GOLD_MARKET = f"{BASE}/data/gold/market"
GOLD_NEWS = f"{BASE}/data/gold/news"
MODELS = f"{BASE}/models"

# Configuration
INSTRUMENT = "spx500"
PREDICTION_HORIZON = 30  # minutes
FINBERT_WINDOW = 60  # minutes for news aggregation


def prepare_directories():
    """Create necessary directories for pipeline outputs."""
    dirs = [
        f"{SILVER_MARKET}/technical",
        f"{SILVER_MARKET}/microstructure",
        f"{SILVER_MARKET}/volatility",
        f"{SILVER_NEWS}/sentiment",
        f"{GOLD_MARKET}/features",
        f"{GOLD_MARKET}/labels",
        f"{GOLD_NEWS}/signals",
        MODELS,
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"[prepare_directories] Created: {dir_path}")

    print("[prepare_directories] All directories ready")


def check_data_availability():
    """Verify that required input data exists."""
    market_file = f"{BRONZE_MARKET}/spx500_usd_m1_5years.ndjson"
    news_dir = f"{BRONZE_NEWS}/historical_5year"

    if not os.path.exists(market_file):
        raise FileNotFoundError(f"Market data not found: {market_file}")

    if not os.path.exists(news_dir):
        raise FileNotFoundError(f"News data not found: {news_dir}")

    news_count = len([f for f in os.listdir(news_dir) if f.endswith('.json')])
    print(f"[check_data] Market data: {market_file} ✓")
    print(f"[check_data] News articles: {news_count} ✓")


# Default arguments for all tasks
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Create DAG
with DAG(
    dag_id="sp500_ml_pipeline_finbert",
    default_args=default_args,
    description="Complete S&P 500 ML pipeline with FinBERT news analysis",
    schedule="0 2 * * *",  # Daily at 2 AM UTC
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["ml", "sp500", "finbert", "production"],
    max_active_runs=1,
) as dag:

    # ========================================================================
    # Stage 0: Preparation
    # ========================================================================

    prepare_dirs = PythonOperator(
        task_id="prepare_directories",
        python_callable=prepare_directories,
    )

    check_data = PythonOperator(
        task_id="check_data_availability",
        python_callable=check_data_availability,
    )

    # ========================================================================
    # Stage 1: Bronze → Silver (Market Features)
    # ========================================================================

    with TaskGroup("market_silver", tooltip="Process market data to silver layer") as market_silver:

        technical_features = BashOperator(
            task_id="technical_features",
            bash_command=f"""
            python {SRC}/data_pipelines/silver/market_technical_processor.py \
                --input {BRONZE_MARKET}/spx500_usd_m1_5years.ndjson \
                --output {SILVER_MARKET}/technical/{INSTRUMENT}_technical.csv
            """,
        )

        microstructure_features = BashOperator(
            task_id="microstructure_features",
            bash_command=f"""
            python {SRC}/data_pipelines/silver/market_microstructure_processor.py \
                --input {BRONZE_MARKET}/spx500_usd_m1_5years.ndjson \
                --output {SILVER_MARKET}/microstructure/{INSTRUMENT}_microstructure.csv
            """,
        )

        volatility_features = BashOperator(
            task_id="volatility_features",
            bash_command=f"""
            python {SRC}/data_pipelines/silver/market_volatility_processor.py \
                --input {BRONZE_MARKET}/spx500_usd_m1_5years.ndjson \
                --output {SILVER_MARKET}/volatility/{INSTRUMENT}_volatility.csv
            """,
        )

        # Run market features in parallel
        [technical_features, microstructure_features, volatility_features]

    # ========================================================================
    # Stage 2: Bronze → Silver (News Sentiment)
    # ========================================================================

    news_silver = BashOperator(
        task_id="news_sentiment_silver",
        bash_command=f"""
        python {SRC}/data_pipelines/silver/news_sentiment_processor.py \
            --input-dir {BRONZE_NEWS} \
            --output {SILVER_NEWS}/sentiment/{INSTRUMENT}_sentiment.csv
        """,
    )

    # ========================================================================
    # Stage 3: Silver → Gold (Market Features)
    # ========================================================================

    market_gold = BashOperator(
        task_id="market_gold_builder",
        bash_command=f"""
        python {SRC}/data_pipelines/gold/market_gold_builder.py \
            --technical {SILVER_MARKET}/technical/{INSTRUMENT}_technical.csv \
            --microstructure {SILVER_MARKET}/microstructure/{INSTRUMENT}_microstructure.csv \
            --volatility {SILVER_MARKET}/volatility/{INSTRUMENT}_volatility.csv \
            --output {GOLD_MARKET}/features/{INSTRUMENT}_features.csv
        """,
    )

    # ========================================================================
    # Stage 4: Silver → Gold (News with FinBERT) ← NEW!
    # ========================================================================

    news_gold_finbert = BashOperator(
        task_id="news_gold_finbert",
        bash_command=f"""
        python {SRC}/data_pipelines/gold/news_signal_builder.py \
            --silver-sentiment {SILVER_NEWS}/sentiment/{INSTRUMENT}_sentiment.csv \
            --bronze-news {BRONZE_NEWS} \
            --output {GOLD_NEWS}/signals/{INSTRUMENT}_news_signals.csv \
            --window {FINBERT_WINDOW}
        """,
        # FinBERT processing can take longer
        execution_timeout=timedelta(hours=2),
    )

    # ========================================================================
    # Stage 5: Generate Prediction Labels
    # ========================================================================

    generate_labels = BashOperator(
        task_id="generate_labels",
        bash_command=f"""
        python {SRC}/data_pipelines/gold/label_generator.py \
            --input {GOLD_MARKET}/features/{INSTRUMENT}_features.csv \
            --output {GOLD_MARKET}/labels/{INSTRUMENT}_labels_{PREDICTION_HORIZON}min.csv \
            --horizon {PREDICTION_HORIZON}
        """,
    )

    # ========================================================================
    # Stage 6: Train Model
    # ========================================================================

    train_model = BashOperator(
        task_id="train_model_xgboost",
        bash_command=f"""
        python {SRC}/training/xgboost_training_pipeline.py \
            --market-features {GOLD_MARKET}/features/{INSTRUMENT}_features.csv \
            --news-signals {GOLD_NEWS}/signals/{INSTRUMENT}_news_signals.csv \
            --prediction-horizon {PREDICTION_HORIZON} \
            --task classification \
            --output-dir {MODELS}
        """,
        execution_timeout=timedelta(hours=1),
    )

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # Preparation
    prepare_dirs >> check_data

    # Silver layers (can run in parallel)
    check_data >> [market_silver, news_silver]

    # Gold layers (market gold depends on all market silver tasks)
    market_silver >> market_gold

    # News gold depends on news silver
    news_silver >> news_gold_finbert

    # Labels depend on market gold
    market_gold >> generate_labels

    # Training depends on labels + news gold
    [generate_labels, news_gold_finbert] >> train_model


# ============================================================================
# DAG Documentation
# ============================================================================

dag.doc_md = """
# S&P 500 ML Pipeline with FinBERT

## Overview
This DAG orchestrates the complete machine learning pipeline for S&P 500 price prediction,
including FinBERT-powered financial sentiment analysis.

## Pipeline Stages

### Stage 1: Market Silver Layer
- **technical_features**: RSI, MACD, Bollinger Bands, SMAs, EMAs
- **microstructure_features**: Spreads, order flow, price impact
- **volatility_features**: GK, Parkinson, Rogers-Satchell, Yang-Zhang

### Stage 2: News Silver Layer
- **news_sentiment_silver**: TextBlob + lexicon-based sentiment (fast preprocessing)

### Stage 3: Market Gold Layer
- **market_gold_builder**: Merge all market features into training-ready format

### Stage 4: News Gold Layer (FinBERT)
- **news_gold_finbert**: Transform sentiment → trading signals using FinBERT
- Model: ProsusAI/finbert (financial domain transformer)
- Output: Aggregated 60-min trading signals (buy/sell/hold)
- Processing time: ~1-2 min per 1000 articles

### Stage 5: Label Generation
- **generate_labels**: Create prediction labels (30-min horizon)

### Stage 6: Model Training
- **train_model_xgboost**: Train XGBoost classifier with news features
- Features: 64 market + 6 news = 70 total
- Output: Trained model + metrics + feature importance

## Monitoring
- Check Airflow logs for each task
- FinBERT stage shows progress bar for article processing
- Final model metrics saved to `{MODELS}/xgboost_*_metrics.json`

## Configuration
Edit these variables in the DAG file:
- `PREDICTION_HORIZON`: Minutes ahead to predict (default: 30)
- `FINBERT_WINDOW`: News aggregation window in minutes (default: 60)
- `INSTRUMENT`: Trading instrument (default: spx500)

## Requirements
- Market data: `{BRONZE_MARKET}/spx500_usd_m1_5years.ndjson`
- News data: `{BRONZE_NEWS}/**/*.json`
- FinBERT model pre-downloaded in Docker image
- Memory: 2-3 GB for FinBERT processing

## Troubleshooting
- **FinBERT timeout**: Increase `execution_timeout` in `news_gold_finbert` task
- **Out of memory**: Reduce batch size or process fewer articles
- **Model not found**: Check FinBERT pre-download in ETL Dockerfile

## Performance
- Market features: ~1-2 minutes
- News sentiment: ~10 seconds (12K articles)
- FinBERT gold: ~10-15 minutes (12K articles, CPU)
- Training: ~30-60 seconds
- **Total**: ~15-20 minutes end-to-end
"""
