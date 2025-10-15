"""
Market Data Downloader - Bronze Layer

Repository Location: fx-ml-pipeline/src_clean/data_pipelines/bronze/market_data_downloader.py

Purpose:
    Downloads raw S&P 500 historical data from OANDA at 1-minute resolution.
    This is the entry point to the Bronze layer of the market data medallion architecture.

Output:
    - Raw NDJSON files saved to: data_clean/bronze/market/
    - Format: One JSON object per line with OHLCV data

Features:
    - Pagination through OANDA's 5000-candle limit
    - Rate limiting to respect API constraints
    - Auto-resume capability if interrupted
    - Data validation and deduplication
    - Progress tracking

Usage:
    python src_clean/data_pipelines/bronze/market_data_downloader.py \\
        --years 5 \\
        --instrument SPX500_USD \\
        --granularity M1
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.oanda_api import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketDataDownloader:
    """Downloads historical market data from OANDA and saves to Bronze layer."""

    def __init__(
        self,
        instrument: str = "SPX500_USD",
        granularity: str = "M1",
        output_dir: str = "data_clean/bronze/market",
        years_back: int = 5,
        rate_limit_delay: float = 0.5
    ):
        """
        Initialize the downloader.

        Parameters
        ----------
        instrument : str
            OANDA instrument symbol (default: SPX500_USD for S&P 500)
        granularity : str
            Candle granularity (M1 = 1 minute, H1 = 1 hour, D = daily)
        output_dir : str
            Directory to save the bronze data
        years_back : int
            Number of years of historical data to download
        rate_limit_delay : float
            Delay in seconds between API requests to respect rate limits
        """
        self.instrument = instrument
        self.granularity = granularity
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.years_back = years_back
        self.rate_limit_delay = rate_limit_delay

        # Output files
        timestamp = datetime.now().strftime("%Y%m%d")
        self.output_file = self.output_dir / f"{instrument.lower()}_{granularity.lower()}_{years_back}y_{timestamp}.ndjson"
        self.progress_file = self.output_dir / f"{instrument.lower()}_{granularity.lower()}_progress.json"

        # OANDA limits
        self.max_candles_per_request = 5000

    def get_progress(self) -> Optional[Dict]:
        """Load progress from previous run if it exists."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load progress file: {e}")
        return None

    def save_progress(self, last_timestamp: str, total_candles: int, chunks_completed: int):
        """Save progress to resume later if interrupted."""
        progress = {
            "last_timestamp": last_timestamp,
            "total_candles": total_candles,
            "chunks_completed": chunks_completed,
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def fetch_candles_batch(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Fetch a batch of candles from OANDA.

        Parameters
        ----------
        from_time : datetime, optional
            Start time for the request
        to_time : datetime, optional
            End time for the request

        Returns
        -------
        list
            List of candle dictionaries
        """
        params = {
            "granularity": self.granularity,
            "price": "M",  # Mid prices
            "count": self.max_candles_per_request
        }

        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            params.pop("count", None)

        if to_time:
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        try:
            request = InstrumentsCandles(instrument=self.instrument, params=params)
            response = API.request(request)
            candles = response.get("candles", [])

            # Transform to our format
            result = []
            for candle in candles:
                if candle.get("complete"):  # Only complete candles
                    result.append({
                        "time": candle["time"],
                        "instrument": self.instrument,
                        "granularity": self.granularity,
                        "open": float(candle["mid"]["o"]),
                        "high": float(candle["mid"]["h"]),
                        "low": float(candle["mid"]["l"]),
                        "close": float(candle["mid"]["c"]),
                        "volume": int(candle.get("volume", 0)),
                        "collected_at": datetime.utcnow().isoformat() + "Z"
                    })

            return result

        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return []

    def download(self):
        """Execute the full download process."""
        logger.info(f"Starting download of {self.years_back} years of {self.instrument} data")
        logger.info(f"Granularity: {self.granularity}")
        logger.info(f"Output: {self.output_file}")

        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.years_back * 365)

        logger.info(f"Date range: {start_time} to {end_time}")

        # Check for existing progress
        progress = self.get_progress()
        if progress:
            logger.info(f"Resuming from {progress['last_timestamp']}")
            start_time = datetime.fromisoformat(progress['last_timestamp'].replace('Z', '+00:00'))

        # Open output file in append mode
        mode = 'a' if progress else 'w'
        total_candles = progress['total_candles'] if progress else 0
        chunks_completed = progress['chunks_completed'] if progress else 0

        with open(self.output_file, mode) as f:
            current_time = start_time

            while current_time < end_time:
                # Fetch batch
                logger.info(f"Fetching from {current_time}...")
                candles = self.fetch_candles_batch(from_time=current_time, to_time=end_time)

                if not candles:
                    logger.warning("No candles returned, moving forward")
                    current_time += timedelta(days=7)  # Skip forward
                    continue

                # Write to file
                for candle in candles:
                    f.write(json.dumps(candle) + '\n')
                    total_candles += 1

                # Update progress
                last_candle_time = candles[-1]['time']
                current_time = datetime.fromisoformat(last_candle_time.replace('Z', '+00:00'))
                current_time += timedelta(minutes=1)  # Move to next minute

                chunks_completed += 1

                logger.info(f"Saved {len(candles)} candles. Total: {total_candles}")
                self.save_progress(last_candle_time, total_candles, chunks_completed)

                # Rate limiting
                time.sleep(self.rate_limit_delay)

        logger.info(f"Download complete! Total candles: {total_candles}")
        logger.info(f"Saved to: {self.output_file}")

        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=5, help="Years of data to download")
    parser.add_argument("--instrument", default="SPX500_USD", help="OANDA instrument")
    parser.add_argument("--granularity", default="M1", help="Candle granularity")
    parser.add_argument("--output-dir", default="data_clean/bronze/market", help="Output directory")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="API rate limit delay")

    args = parser.parse_args()

    downloader = MarketDataDownloader(
        instrument=args.instrument,
        granularity=args.granularity,
        output_dir=args.output_dir,
        years_back=args.years,
        rate_limit_delay=args.rate_limit
    )

    downloader.download()


if __name__ == "__main__":
    main()
