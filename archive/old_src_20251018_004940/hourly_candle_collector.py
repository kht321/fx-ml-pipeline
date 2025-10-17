"""Hourly candle collector for 2025 live data streaming.

This module streams live USD_SGD candles from OANDA and saves them to Bronze layer
in NDJSON format for real-time training data collection.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging

from oanda_api import API, fetch_candles
from oandapyV20.endpoints.instruments import InstrumentsCandles

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HourlyCandleCollector:
    """Collects hourly USD_SGD candles and saves to Bronze layer."""

    def __init__(self, instrument: str = "USD_SGD", output_dir: str = "data/bronze/prices"):
        self.instrument = instrument
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{instrument.lower()}_hourly_2025.ndjson"

    def get_latest_candle_time(self) -> Optional[datetime]:
        """Get the timestamp of the last collected candle."""
        if not self.output_file.exists():
            return None

        try:
            with open(self.output_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        data = json.loads(last_line)
                        return datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Error reading last candle time: {e}")

        return None

    def fetch_hourly_candles(self, from_time: Optional[datetime] = None, count: int = 500) -> list:
        """Fetch hourly candles from OANDA."""
        params = {
            "granularity": "H1",  # Hourly candles
            "count": count,
            "price": "MBA"  # Mid, Bid, Ask prices
        }

        # If we have a from_time, use it to get only new candles
        if from_time:
            # Add 1 hour to avoid duplicating the last candle
            from_time += timedelta(hours=1)
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            # Remove count when using from parameter
            params.pop("count", None)

        request = InstrumentsCandles(instrument=self.instrument, params=params)
        response = API.request(request)

        return response.get("candles", [])

    def save_candles(self, candles: list) -> int:
        """Save candles to NDJSON file."""
        saved_count = 0

        with open(self.output_file, 'a') as f:
            for candle in candles:
                # Only save complete candles
                if candle.get("complete", False):
                    # Transform to our Bronze layer format
                    bronze_candle = {
                        "time": candle["time"],
                        "instrument": self.instrument,
                        "granularity": "H1",
                        "open": float(candle["mid"]["o"]),
                        "high": float(candle["mid"]["h"]),
                        "low": float(candle["mid"]["l"]),
                        "close": float(candle["mid"]["c"]),
                        "volume": int(candle["volume"]),
                        "bid_open": float(candle["bid"]["o"]),
                        "bid_high": float(candle["bid"]["h"]),
                        "bid_low": float(candle["bid"]["l"]),
                        "bid_close": float(candle["bid"]["c"]),
                        "ask_open": float(candle["ask"]["o"]),
                        "ask_high": float(candle["ask"]["h"]),
                        "ask_low": float(candle["ask"]["l"]),
                        "ask_close": float(candle["ask"]["c"]),
                        "spread": float(candle["ask"]["c"]) - float(candle["bid"]["c"]),
                        "collected_at": datetime.utcnow().isoformat() + "Z"
                    }

                    f.write(json.dumps(bronze_candle) + '\n')
                    saved_count += 1

        return saved_count

    def backfill_missing_data(self):
        """Backfill any missing hourly data from the start of 2025."""
        logger.info("Checking for missing data to backfill...")

        # Get last collected candle
        last_time = self.get_latest_candle_time()

        if last_time is None:
            # No data yet, start from beginning of 2025
            start_time = datetime(2025, 1, 1)
            logger.info("No existing data found. Starting from 2025-01-01")
        else:
            start_time = last_time
            logger.info(f"Last candle: {last_time}. Checking for newer data...")

        # Fetch candles from start_time to now
        candles = self.fetch_hourly_candles(from_time=start_time)

        if candles:
            saved = self.save_candles(candles)
            logger.info(f"Backfilled {saved} new hourly candles")
        else:
            logger.info("No new candles to backfill")

    def start_live_collection(self, check_interval: int = 3600):
        """Start collecting live hourly candles."""
        logger.info(f"Starting live hourly candle collection for {self.instrument}")
        logger.info(f"Output file: {self.output_file}")

        # First, backfill any missing data
        self.backfill_missing_data()

        logger.info(f"Starting live collection (checking every {check_interval} seconds)")

        while True:
            try:
                # Check for new candles
                last_time = self.get_latest_candle_time()
                candles = self.fetch_hourly_candles(from_time=last_time)

                if candles:
                    saved = self.save_candles(candles)
                    if saved > 0:
                        logger.info(f"Collected {saved} new hourly candles")

                # Wait for next check
                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("Collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during collection: {e}")
                logger.info("Retrying in 60 seconds...")
                time.sleep(60)


def main():
    """CLI entry point for hourly candle collection."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect hourly USD_SGD candles for 2025")
    parser.add_argument("--instrument", default="USD_SGD", help="Trading instrument")
    parser.add_argument("--output-dir", default="data/bronze/prices", help="Output directory")
    parser.add_argument("--backfill-only", action="store_true", help="Only backfill, don't start live collection")
    parser.add_argument("--check-interval", type=int, default=3600, help="Check interval in seconds (default: 1 hour)")

    args = parser.parse_args()

    collector = HourlyCandleCollector(
        instrument=args.instrument,
        output_dir=args.output_dir
    )

    if args.backfill_only:
        collector.backfill_missing_data()
    else:
        collector.start_live_collection(check_interval=args.check_interval)


if __name__ == "__main__":
    main()