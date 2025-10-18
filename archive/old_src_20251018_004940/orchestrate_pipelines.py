"""Orchestrate both Market and News medallion pipelines.

This script coordinates the execution of both medallion pipelines, ensuring
proper sequencing, error handling, and monitoring of the dual-pipeline architecture.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for pipeline orchestration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["market", "news", "combined", "all"],
        default="all",
        help="Which pipeline(s) to run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/combined_features.yaml"),
        help="Configuration file for pipeline coordination",
    )
    parser.add_argument(
        "--bronze-to-silver",
        action="store_true",
        help="Run Bronze to Silver layer processing",
    )
    parser.add_argument(
        "--silver-to-gold",
        action="store_true",
        help="Run Silver to Gold layer processing",
    )
    parser.add_argument(
        "--train-models",
        action="store_true",
        help="Train combined models using Gold layer data",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run in continuous mode (daemon-like operation)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval in seconds for continuous mode (default: 5 minutes)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    """Load pipeline configuration from YAML file."""
    try:
        with config_path.open('r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        log(f"ERROR", f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def log(level: str, message: str) -> None:
    """Structured logging for pipeline orchestration."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] [{level}] [orchestrate] {message}")


def run_command(cmd: List[str], description: str, cwd: Path = None) -> bool:
    """Run a subprocess command with error handling."""
    log("INFO", f"Starting: {description}")
    log("DEBUG", f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            log("INFO", f"Completed: {description}")
            if result.stdout.strip():
                log("DEBUG", f"Output: {result.stdout.strip()}")
            return True
        else:
            log("ERROR", f"Failed: {description}")
            log("ERROR", f"Error: {result.stderr.strip()}")
            return False

    except subprocess.TimeoutExpired:
        log("ERROR", f"Timeout: {description}")
        return False
    except Exception as e:
        log("ERROR", f"Exception in {description}: {e}")
        return False


def check_data_availability(paths: List[Path]) -> Dict[str, bool]:
    """Check if required data files are available."""
    availability = {}
    for path in paths:
        availability[str(path)] = path.exists() and path.stat().st_size > 0
    return availability


def run_market_pipeline(config: Dict, bronze_to_silver: bool = True, silver_to_gold: bool = True) -> bool:
    """Execute the market data medallion pipeline."""
    log("INFO", "=== Starting Market Pipeline ===")

    success = True
    market_config = config.get('pipeline_coordination', {}).get('market_pipeline', {})

    # Bronze to Silver: Process price ticks into features
    if bronze_to_silver:
        # Check for bronze data
        bronze_path = Path(market_config.get('bronze_input', 'data/market/bronze/prices/'))

        if not any(bronze_path.glob('*.ndjson')):
            log("WARNING", "No bronze price data found - skipping market feature processing")
        else:
            cmd = [
                "python", "src/build_market_features.py",
                "--input", str(bronze_path / "usd_sgd_stream.ndjson"),
                "--output-technical", "data/market/silver/technical_features/sgd_vs_majors.csv",
                "--output-microstructure", "data/market/silver/microstructure/depth_features.csv",
                "--output-volatility", "data/market/silver/volatility/risk_metrics.csv",
                "--flush-interval", "100",
                "--log-every", "100"
            ]

            success &= run_command(cmd, "Market Bronze → Silver processing")

    # Silver to Gold: Consolidate market features
    if silver_to_gold and success:
        cmd = [
            "python", "src/build_market_gold.py",
            "--technical-features", "data/market/silver/technical_features/sgd_vs_majors.csv",
            "--microstructure-features", "data/market/silver/microstructure/depth_features.csv",
            "--volatility-features", "data/market/silver/volatility/risk_metrics.csv",
            "--output", market_config.get('gold_output', 'data/market/gold/training/market_features.csv'),
            "--feature-selection", "all"
        ]

        success &= run_command(cmd, "Market Silver → Gold consolidation")

    log("INFO", f"Market pipeline completed: {'SUCCESS' if success else 'FAILURE'}")
    return success


def run_news_pipeline(config: Dict, bronze_to_silver: bool = True, silver_to_gold: bool = True, use_fingpt: bool = True) -> bool:
    """Execute the news medallion pipeline with FinGPT."""
    log("INFO", "=== Starting News Pipeline ===")

    success = True
    news_config = config.get('pipeline_coordination', {}).get('news_pipeline', {})

    # Bronze to Silver: Process news articles into sentiment features
    if bronze_to_silver:
        # Check for bronze news data
        bronze_path = Path(news_config.get('bronze_input', 'data/news/bronze/raw_articles/'))

        if not any(bronze_path.glob('*')):
            log("WARNING", "No bronze news data found - skipping news feature processing")
        else:
            cmd = [
                "python", "src/build_news_features.py",
                "--input-dir", str(bronze_path),
                "--output-sentiment", "data/news/silver/sentiment_scores/sentiment_features.csv",
                "--output-entities", "data/news/silver/entity_mentions/entity_features.csv",
                "--output-topics", "data/news/silver/topic_signals/topic_features.csv",
                "--batch-size", "10"
            ]

            if use_fingpt:
                cmd.extend(["--use-fingpt", "--fingpt-model", "FinGPT/fingpt-sentiment_llama2-7b_lora"])

            success &= run_command(cmd, f"News Bronze → Silver processing ({'FinGPT' if use_fingpt else 'Lexicon'} mode)")

    # Silver to Gold: Create trading signals
    if silver_to_gold and success:
        cmd = [
            "python", "src/build_news_gold.py",
            "--sentiment-features", "data/news/silver/sentiment_scores/sentiment_features.csv",
            "--entity-features", "data/news/silver/entity_mentions/entity_features.csv",
            "--topic-features", "data/news/silver/topic_signals/topic_features.csv",
            "--output", news_config.get('gold_output', 'data/news/gold/news_signals/trading_signals.csv'),
            "--lookback-hours", "24",
            "--min-confidence", "0.3"
        ]

        success &= run_command(cmd, "News Silver → Gold signal generation")

    log("INFO", f"News pipeline completed: {'SUCCESS' if success else 'FAILURE'}")
    return success


def run_combined_modeling(config: Dict) -> bool:
    """Train combined models using both market and news Gold layer data."""
    log("INFO", "=== Starting Combined Modeling ===")

    # Check data availability
    market_gold = Path("data/market/gold/training/market_features.csv")
    news_gold = Path("data/news/gold/news_signals/trading_signals.csv")

    availability = check_data_availability([market_gold, news_gold])

    if not availability[str(market_gold)]:
        log("ERROR", "Market Gold data not available - cannot train combined models")
        return False

    if not availability[str(news_gold)]:
        log("WARNING", "News Gold data not available - training market-only models")

    # Configure models to train
    model_config = config.get('combined_modeling', {}).get('models', {})
    models_to_train = [name for name, config in model_config.items() if config.get('enabled', True)]

    cmd = [
        "python", "src/train_combined_model.py",
        "--market-features", str(market_gold),
        "--news-signals", str(news_gold),
        "--output-dir", "data/combined/models",
        "--focus-currency", config.get('combined_modeling', {}).get('target_currency', 'USD_SGD'),
        "--models"] + models_to_train + [
        "--cross-validation"
    ]

    success = run_command(cmd, "Combined model training")

    log("INFO", f"Combined modeling completed: {'SUCCESS' if success else 'FAILURE'}")
    return success


def pipeline_health_check() -> Dict[str, bool]:
    """Check the health status of both pipelines."""
    health = {}

    # Check data freshness
    paths_to_check = [
        ("market_bronze", "data/market/bronze/prices/usd_sgd_stream.ndjson"),
        ("market_silver", "data/market/silver/technical_features/sgd_vs_majors.csv"),
        ("market_gold", "data/market/gold/training/market_features.csv"),
        ("news_bronze", "data/news/bronze/raw_articles/"),
        ("news_silver", "data/news/silver/sentiment_scores/sentiment_features.csv"),
        ("news_gold", "data/news/gold/news_signals/trading_signals.csv"),
        ("combined_models", "data/combined/models/")
    ]

    for name, path_str in paths_to_check:
        path = Path(path_str)
        if path.is_dir():
            # Check if directory has recent files
            recent_files = [f for f in path.glob('*') if f.is_file() and
                          (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)) < timedelta(hours=24)]
            health[name] = len(recent_files) > 0
        else:
            # Check if file exists and is recent
            health[name] = (path.exists() and
                          (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)) < timedelta(hours=24))

    return health


def run_single_iteration(config: Dict, mode: str, bronze_to_silver: bool,
                        silver_to_gold: bool, train_models: bool) -> bool:
    """Run a single iteration of the specified pipeline(s)."""

    overall_success = True

    if mode in ["market", "all"]:
        success = run_market_pipeline(config, bronze_to_silver, silver_to_gold)
        overall_success &= success

    if mode in ["news", "all"]:
        success = run_news_pipeline(config, bronze_to_silver, silver_to_gold, use_fingpt=True)
        overall_success &= success

    if mode in ["combined", "all"] and train_models:
        success = run_combined_modeling(config)
        overall_success &= success

    # Health check
    health = pipeline_health_check()
    healthy_components = sum(health.values())
    total_components = len(health)

    log("INFO", f"Pipeline health: {healthy_components}/{total_components} components healthy")
    for component, is_healthy in health.items():
        status = "✓" if is_healthy else "✗"
        log("INFO", f"  {status} {component}")

    return overall_success


def main() -> None:
    """Main orchestration function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    log("INFO", f"Starting pipeline orchestration (mode: {args.mode})")
    log("INFO", f"Configuration loaded from: {args.config}")

    if args.continuous:
        log("INFO", f"Running in continuous mode (interval: {args.interval}s)")

        try:
            iteration = 0
            while True:
                iteration += 1
                log("INFO", f"=== Iteration {iteration} ===")

                success = run_single_iteration(
                    config, args.mode, args.bronze_to_silver,
                    args.silver_to_gold, args.train_models
                )

                if success:
                    log("INFO", f"Iteration {iteration} completed successfully")
                else:
                    log("WARNING", f"Iteration {iteration} had failures")

                log("INFO", f"Sleeping for {args.interval} seconds...")
                time.sleep(args.interval)

        except KeyboardInterrupt:
            log("INFO", "Continuous mode interrupted by user")
        except Exception as e:
            log("ERROR", f"Continuous mode failed: {e}")
            sys.exit(1)

    else:
        # Single run
        success = run_single_iteration(
            config, args.mode, args.bronze_to_silver,
            args.silver_to_gold, args.train_models
        )

        if success:
            log("INFO", "Pipeline orchestration completed successfully")
            sys.exit(0)
        else:
            log("ERROR", "Pipeline orchestration failed")
            sys.exit(1)


if __name__ == "__main__":
    main()