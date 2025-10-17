"""2025 Live Data Collection Pipeline.

Orchestrates collection of both market data (OANDA) and news data for real-time
training dataset generation throughout 2025.
"""

import asyncio
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import time

from hourly_candle_collector import HourlyCandleCollector
from news_scraper import NewsScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollectionPipeline:
    """Orchestrates live data collection for 2025."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Initialize collectors
        self.candle_collector = HourlyCandleCollector(
            instrument="USD_SGD",
            output_dir=str(self.data_dir / "bronze" / "prices")
        )
        self.news_scraper = NewsScraper(
            output_dir=str(self.data_dir / "bronze" / "news")
        )

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.running = False

    async def start_market_collection(self):
        """Start market data collection in background thread."""
        logger.info("Starting market data collection...")

        def run_market_collection():
            try:
                # First backfill any missing data
                self.candle_collector.backfill_missing_data()

                # Then start live collection with 10-minute checks
                # (OANDA hourly candles complete a few minutes after the hour)
                while self.running:
                    try:
                        last_time = self.candle_collector.get_latest_candle_time()
                        candles = self.candle_collector.fetch_hourly_candles(from_time=last_time)

                        if candles:
                            saved = self.candle_collector.save_candles(candles)
                            if saved > 0:
                                logger.info(f"Market: Collected {saved} new hourly candles")

                        # Check every 10 minutes for new completed candles
                        for _ in range(60):  # 60 * 10 seconds = 10 minutes
                            if not self.running:
                                break
                            time.sleep(10)

                    except Exception as e:
                        logger.error(f"Market collection error: {e}")
                        time.sleep(60)  # Wait 1 minute on error

            except Exception as e:
                logger.error(f"Market collection thread failed: {e}")

        # Run in thread pool
        future = self.executor.submit(run_market_collection)
        return future

    async def start_news_collection(self):
        """Start news collection in background."""
        logger.info("Starting news collection...")

        try:
            # Start live news collection (checks every 30 minutes)
            while self.running:
                try:
                    results = await self.news_scraper.run_collection_cycle()
                    total_articles = sum(results.values())
                    if total_articles > 0:
                        logger.info(f"News: Collected {total_articles} SGD-relevant articles")

                    # Wait 30 minutes between news collection cycles
                    for _ in range(180):  # 180 * 10 seconds = 30 minutes
                        if not self.running:
                            break
                        await asyncio.sleep(10)

                except Exception as e:
                    logger.error(f"News collection error: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes on error

        except Exception as e:
            logger.error(f"News collection thread failed: {e}")

    async def monitor_pipeline(self):
        """Monitor the overall pipeline health."""
        logger.info("Starting pipeline monitoring...")

        while self.running:
            # Log pipeline status every hour
            try:
                # Check market data freshness
                last_candle_time = self.candle_collector.get_latest_candle_time()
                if last_candle_time:
                    hours_since_last = (datetime.utcnow() - last_candle_time.replace(tzinfo=None)).total_seconds() / 3600
                    logger.info(f"Pipeline Status: Last candle {hours_since_last:.1f} hours ago")
                else:
                    logger.warning("Pipeline Status: No market data collected yet")

                # Check data directory sizes
                market_file = self.data_dir / "bronze" / "prices" / "usd_sgd_hourly_2025.ndjson"
                news_file = self.data_dir / "bronze" / "news" / "financial_news_2025.ndjson"

                if market_file.exists():
                    size_mb = market_file.stat().st_size / (1024 * 1024)
                    logger.info(f"Market data file: {size_mb:.1f} MB")

                if news_file.exists():
                    size_mb = news_file.stat().st_size / (1024 * 1024)
                    logger.info(f"News data file: {size_mb:.1f} MB")

            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            # Wait 1 hour
            for _ in range(360):  # 360 * 10 seconds = 1 hour
                if not self.running:
                    break
                await asyncio.sleep(10)

    async def run(self):
        """Run the complete data collection pipeline."""
        logger.info("=== Starting 2025 Live Data Collection Pipeline ===")
        logger.info(f"Data directory: {self.data_dir.absolute()}")

        self.running = True

        try:
            # Start all collection tasks
            tasks = [
                self.start_market_collection(),
                asyncio.create_task(self.start_news_collection()),
                asyncio.create_task(self.monitor_pipeline())
            ]

            # Wait for all tasks (they run indefinitely)
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            logger.info("Pipeline shutting down...")
            self.running = False
            self.executor.shutdown(wait=True)

    def status(self):
        """Get current pipeline status."""
        status = {
            "pipeline_running": self.running,
            "data_directory": str(self.data_dir.absolute()),
            "market_data": {},
            "news_data": {}
        }

        # Market data status
        market_file = self.data_dir / "bronze" / "prices" / "usd_sgd_hourly_2025.ndjson"
        if market_file.exists():
            last_candle_time = self.candle_collector.get_latest_candle_time()
            status["market_data"] = {
                "file_exists": True,
                "file_size_mb": market_file.stat().st_size / (1024 * 1024),
                "last_candle": last_candle_time.isoformat() if last_candle_time else None
            }
        else:
            status["market_data"] = {"file_exists": False}

        return status


async def main():
    """CLI entry point for the data collection pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="2025 Live Data Collection Pipeline")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--status", action="store_true", help="Show pipeline status and exit")

    args = parser.parse_args()

    pipeline = DataCollectionPipeline(data_dir=args.data_dir)

    if args.status:
        # Just show status
        status = pipeline.status()
        print("=== Pipeline Status ===")
        print(f"Running: {status['pipeline_running']}")
        print(f"Data Dir: {status['data_directory']}")
        print(f"Market Data: {status['market_data']}")
        print(f"News Data: {status['news_data']}")
        return

    try:
        await pipeline.run()
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())