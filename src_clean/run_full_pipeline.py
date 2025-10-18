"""
Full Pipeline Orchestrator

Repository Location: fx-ml-pipeline/src_clean/run_full_pipeline.py

Purpose:
    Orchestrates the complete data pipeline from Bronze → Silver → Gold → Training.
    Runs all processing steps in sequence to prepare training-ready data.

Pipeline Steps:
    1. Bronze → Silver (Market): Technical + Microstructure + Volatility
    2. Bronze → Silver (News): Sentiment + Entities + Topics
    3. Silver → Gold (Market): Merge all market features
    4. Silver → Gold (News): Build trading signals
    5. Gold: Generate prediction labels
    6. Train: XGBoost model training

News Sources Supported:
    - Automatically processes ALL news sources in bronze/news/:
      * bronze/news/*.json (original RSS scraper)
      * bronze/news/hybrid/*.json (hybrid scraper - GDELT, Finnhub, etc.)
      * Merges and deduplicates all articles

Usage:
    # Full pipeline with hybrid news scraper
    python src_clean/run_full_pipeline.py \\
        --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \\
        --bronze-news data_clean/bronze/news \\
        --output-dir data_clean

    # The pipeline automatically finds and processes:
    #   - data_clean/bronze/news/*.json (RSS feeds)
    #   - data_clean/bronze/news/hybrid/*.json (GDELT + APIs)

    # Skip specific stages
    python src_clean/run_full_pipeline.py \\
        --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \\
        --skip-news \\
        --skip-training
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full ML pipeline."""

    def __init__(
        self,
        bronze_market: Path,
        bronze_news: Path,
        output_dir: Path,
        instrument: str = "spx500",
        prediction_horizon: int = 30
    ):
        self.bronze_market = bronze_market
        self.bronze_news = bronze_news
        self.output_dir = output_dir
        self.instrument = instrument
        self.prediction_horizon = prediction_horizon

        # Define output paths
        self.silver_technical = output_dir / f"silver/market/technical/{instrument}_technical.csv"
        self.silver_micro = output_dir / f"silver/market/microstructure/{instrument}_microstructure.csv"
        self.silver_vol = output_dir / f"silver/market/volatility/{instrument}_volatility.csv"
        self.silver_sentiment = output_dir / f"silver/news/sentiment/{instrument}_sentiment.csv"

        self.gold_market = output_dir / f"gold/market/features/{instrument}_features.csv"
        self.gold_labels = output_dir / f"gold/market/labels/{instrument}_labels_{prediction_horizon}min.csv"

        self.model_dir = output_dir / "models"

    def run_command(self, cmd: list, description: str):
        """Run a subprocess command."""
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP: {description}")
        logger.info(f"{'='*80}")
        logger.info(f"Command: {' '.join(str(c) for c in cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("✓ Success")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return False

    def process_silver_market(self) -> bool:
        """Process bronze → silver for market data."""
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: BRONZE → SILVER (MARKET)")
        logger.info("="*80)

        # Technical features
        success = self.run_command(
            [
                sys.executable,
                "src_clean/data_pipelines/silver/market_technical_processor.py",
                "--input", str(self.bronze_market),
                "--output", str(self.silver_technical)
            ],
            "Market Technical Features"
        )

        if not success:
            return False

        # Microstructure features
        success = self.run_command(
            [
                sys.executable,
                "src_clean/data_pipelines/silver/market_microstructure_processor.py",
                "--input", str(self.bronze_market),
                "--output", str(self.silver_micro)
            ],
            "Market Microstructure Features"
        )

        if not success:
            return False

        # Volatility features
        success = self.run_command(
            [
                sys.executable,
                "src_clean/data_pipelines/silver/market_volatility_processor.py",
                "--input", str(self.bronze_market),
                "--output", str(self.silver_vol)
            ],
            "Market Volatility Features"
        )

        return success

    def process_silver_news(self) -> bool:
        """Process bronze → silver for news data."""
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: BRONZE → SILVER (NEWS)")
        logger.info("="*80)

        # Sentiment features
        success = self.run_command(
            [
                sys.executable,
                "src_clean/data_pipelines/silver/news_sentiment_processor.py",
                "--input-dir", str(self.bronze_news),
                "--output", str(self.silver_sentiment)
            ],
            "News Sentiment Features"
        )

        return success

    def build_gold_market(self) -> bool:
        """Build gold layer market features."""
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: SILVER → GOLD (MARKET)")
        logger.info("="*80)

        success = self.run_command(
            [
                sys.executable,
                "src_clean/data_pipelines/gold/market_gold_builder.py",
                "--technical", str(self.silver_technical),
                "--microstructure", str(self.silver_micro),
                "--volatility", str(self.silver_vol),
                "--output", str(self.gold_market)
            ],
            "Market Gold Layer"
        )

        return success

    def generate_labels(self) -> bool:
        """Generate prediction labels."""
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: GENERATE LABELS")
        logger.info("="*80)

        success = self.run_command(
            [
                sys.executable,
                "src_clean/data_pipelines/gold/label_generator.py",
                "--input", str(self.gold_market),
                "--output", str(self.gold_labels),
                "--horizon", str(self.prediction_horizon)
            ],
            f"Prediction Labels ({self.prediction_horizon}min)"
        )

        return success

    def train_model(self) -> bool:
        """Train XGBoost model."""
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: TRAIN MODEL")
        logger.info("="*80)

        success = self.run_command(
            [
                sys.executable,
                "src_clean/training/xgboost_training_pipeline.py",
                "--market-features", str(self.gold_market),
                "--prediction-horizon", str(self.prediction_horizon),
                "--task", "classification",
                "--output-dir", str(self.model_dir)
            ],
            "XGBoost Model Training"
        )

        return success

    def run(self, skip_news: bool = False, skip_training: bool = False):
        """Execute full pipeline."""
        start_time = datetime.now()

        logger.info("\n")
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*25 + "FULL PIPELINE EXECUTION" + " "*30 + "║")
        logger.info("╚" + "="*78 + "╝")
        logger.info("")

        # Stage 1: Market silver
        if not self.process_silver_market():
            logger.error("Pipeline failed at market silver stage")
            return False

        # Stage 2: News silver (optional)
        if not skip_news and self.bronze_news.exists():
            if not self.process_silver_news():
                logger.warning("News processing failed, continuing without news features")

        # Stage 3: Market gold
        if not self.build_gold_market():
            logger.error("Pipeline failed at market gold stage")
            return False

        # Stage 4: Labels
        if not self.generate_labels():
            logger.error("Pipeline failed at label generation stage")
            return False

        # Stage 5: Training (optional)
        if not skip_training:
            if not self.train_model():
                logger.error("Pipeline failed at training stage")
                return False

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Gold features: {self.gold_market}")
        logger.info(f"Labels: {self.gold_labels}")
        if not skip_training:
            logger.info(f"Models: {self.model_dir}")
        logger.info("="*80)

        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze-market", type=Path, required=True, help="Bronze market NDJSON")
    parser.add_argument("--bronze-news", type=Path, help="Bronze news directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data_clean"), help="Output directory")
    parser.add_argument("--instrument", default="spx500", help="Instrument name")
    parser.add_argument("--prediction-horizon", type=int, default=30, help="Prediction horizon (minutes)")
    parser.add_argument("--skip-news", action="store_true", help="Skip news processing")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        bronze_market=args.bronze_market,
        bronze_news=args.bronze_news or Path("data_clean/bronze/news"),
        output_dir=args.output_dir,
        instrument=args.instrument,
        prediction_horizon=args.prediction_horizon
    )

    success = orchestrator.run(
        skip_news=args.skip_news,
        skip_training=args.skip_training
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
