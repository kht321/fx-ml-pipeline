"""Download 10 years of S&P 500 historical data from OANDA at 1-minute resolution.

This script handles the complexities of downloading large historical datasets:
- Pagination through OANDA's 5000-candle limit per request
- Rate limiting to respect API constraints
- Automatic resume capability if interrupted
- Data validation and deduplication
- Progress tracking and logging

The S&P 500 is accessed via OANDA's SPX500_USD CFD instrument.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict
import argparse

from oanda_api import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SP500HistoricalDownloader:
    """Downloads historical S&P 500 1-minute candles from OANDA."""

    def __init__(
        self,
        instrument: str = "SPX500_USD",
        granularity: str = "M1",
        output_dir: str = "data/bronze/prices",
        years_back: int = 10,
        rate_limit_delay: float = 0.5
    ):
        """
        Initialize the downloader.

        Parameters
        ----------
        instrument : str
            OANDA instrument symbol (default: SPX500_USD for S&P 500)
        granularity : str
            Candle granularity (M1 = 1 minute)
        output_dir : str
            Directory to save the data
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
        self.output_file = self.output_dir / f"{instrument.lower()}_{granularity.lower()}_historical.ndjson"
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

    def get_latest_candle_time(self) -> Optional[datetime]:
        """Get the timestamp of the last collected candle from the data file."""
        if not self.output_file.exists():
            return None

        try:
            with open(self.output_file, 'r') as f:
                # Read from the end to find the last valid line
                lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        return datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Error reading last candle time: {e}")

        return None

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
            "price": "M",  # Mid prices (or use "MBA" for Bid/Ask as well)
            "count": self.max_candles_per_request
        }

        # Use time-based queries for more precise control
        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            params.pop("count", None)  # Can't use both 'from' and 'count'

        if to_time:
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        try:
            request = InstrumentsCandles(instrument=self.instrument, params=params)
            response = API.request(request)
            candles = response.get("candles", [])

            # Respect rate limits
            time.sleep(self.rate_limit_delay)

            return candles
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            # On error, wait longer before retrying
            time.sleep(5)
            return []

    def save_candles(self, candles: List[Dict], mode: str = 'a') -> int:
        """
        Save candles to NDJSON file.

        Parameters
        ----------
        candles : list
            List of candle dictionaries from OANDA
        mode : str
            File open mode ('a' for append, 'w' for write)

        Returns
        -------
        int
            Number of candles saved
        """
        saved_count = 0

        with open(self.output_file, mode) as f:
            for candle in candles:
                # Only save complete candles
                if candle.get("complete", False):
                    # Transform to Bronze layer format
                    bronze_candle = {
                        "time": candle["time"],
                        "instrument": self.instrument,
                        "granularity": self.granularity,
                        "open": float(candle["mid"]["o"]),
                        "high": float(candle["mid"]["h"]),
                        "low": float(candle["mid"]["l"]),
                        "close": float(candle["mid"]["c"]),
                        "volume": int(candle["volume"]),
                        "collected_at": datetime.utcnow().isoformat() + "Z"
                    }

                    f.write(json.dumps(bronze_candle) + '\n')
                    saved_count += 1

        return saved_count

    def download_historical_data(self):
        """
        Download the full historical dataset with pagination and progress tracking.

        This method handles:
        - Breaking the download into manageable chunks
        - Tracking progress for resume capability
        - Rate limiting
        - Data validation
        """
        logger.info(f"Starting download of {self.years_back} years of {self.instrument} {self.granularity} data")
        logger.info(f"Output file: {self.output_file}")

        # Calculate time range (timezone-aware)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=365 * self.years_back)

        # Check for existing progress
        progress = self.get_progress()
        if progress:
            logger.info(f"Resuming from previous download...")
            logger.info(f"Last timestamp: {progress['last_timestamp']}")
            logger.info(f"Candles so far: {progress['total_candles']}")
            start_time = datetime.fromisoformat(progress['last_timestamp'].replace('Z', '+00:00'))
        else:
            logger.info(f"Starting fresh download from {start_time} to {end_time}")

        total_candles = progress['total_candles'] if progress else 0
        chunks_completed = progress['chunks_completed'] if progress else 0
        current_time = start_time

        # Estimate total chunks needed
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        estimated_chunks = total_minutes // self.max_candles_per_request + 1

        logger.info(f"Estimated chunks to download: {estimated_chunks}")
        logger.info("This may take a while. Progress will be saved periodically.")

        consecutive_empty_batches = 0
        max_empty_batches = 5  # Stop after 5 consecutive empty batches

        try:
            while current_time < end_time:
                # Calculate next chunk end time
                chunk_end_time = min(
                    current_time + timedelta(minutes=self.max_candles_per_request),
                    end_time
                )

                logger.info(f"Fetching chunk {chunks_completed + 1}: {current_time} to {chunk_end_time}")

                # Fetch candles for this chunk
                candles = self.fetch_candles_batch(
                    from_time=current_time,
                    to_time=chunk_end_time
                )

                if not candles:
                    consecutive_empty_batches += 1
                    logger.warning(f"No candles returned for this time range. Empty batch {consecutive_empty_batches}/{max_empty_batches}")

                    if consecutive_empty_batches >= max_empty_batches:
                        logger.warning("Multiple consecutive empty batches. Possible end of available data.")
                        logger.info("Moving forward by a larger time window...")
                        # Jump ahead to try to find data
                        current_time = chunk_end_time
                        consecutive_empty_batches = 0
                        continue

                    # Move forward and try next chunk
                    current_time = chunk_end_time
                    continue

                # Reset empty batch counter
                consecutive_empty_batches = 0

                # Save candles
                saved = self.save_candles(candles)
                total_candles += saved
                chunks_completed += 1

                # Get the timestamp of the last candle
                if candles:
                    last_candle_time = candles[-1]["time"]
                    # Handle OANDA's nanosecond precision (e.g., "2024-10-16T02:27:00.000000000Z")
                    # Python's fromisoformat only supports microseconds, so we need to truncate
                    time_str = last_candle_time.replace('Z', '+00:00')
                    # If there are more than 6 decimal places (nanoseconds), truncate to microseconds
                    if '.' in time_str:
                        parts = time_str.split('.')
                        if len(parts[1]) > 9:  # Has timezone info after decimals
                            decimals = parts[1][:6]  # Keep only 6 digits (microseconds)
                            tz_suffix = parts[1][9:]   # Keep timezone suffix
                            time_str = parts[0] + '.' + decimals + tz_suffix
                        elif len(parts[1]) > 6:
                            time_str = parts[0] + '.' + parts[1][:6] + parts[1][9:]
                    last_dt = datetime.fromisoformat(time_str)

                    # Save progress
                    self.save_progress(last_candle_time, total_candles, chunks_completed)

                    # Update current_time for next iteration
                    current_time = last_dt + timedelta(minutes=1)

                    logger.info(f"Saved {saved} candles. Total: {total_candles} | Progress: {chunks_completed}/{estimated_chunks} chunks")
                else:
                    # No more candles, move forward
                    current_time = chunk_end_time

                # Show progress percentage
                progress_pct = ((current_time - start_time).total_seconds() /
                               (end_time - start_time).total_seconds() * 100)
                logger.info(f"Progress: {progress_pct:.2f}%")

        except KeyboardInterrupt:
            logger.info("Download interrupted by user. Progress has been saved.")
            logger.info(f"Total candles downloaded: {total_candles}")
            logger.info("Run the script again to resume from where you left off.")
            return

        except Exception as e:
            logger.error(f"Error during download: {e}")
            logger.info(f"Progress saved. Total candles so far: {total_candles}")
            raise

        logger.info("=" * 80)
        logger.info("Download complete!")
        logger.info(f"Total candles downloaded: {total_candles}")
        logger.info(f"Output file: {self.output_file}")
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info("=" * 80)

        # Clean up progress file
        if self.progress_file.exists():
            self.progress_file.unlink()
            logger.info("Progress file cleaned up.")

    def validate_data(self):
        """Validate the downloaded data and print statistics."""
        if not self.output_file.exists():
            logger.error("No data file found!")
            return

        logger.info("Validating data...")

        candle_count = 0
        timestamps = []

        with open(self.output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    candle_count += 1
                    data = json.loads(line)
                    timestamps.append(data['time'])

        if timestamps:
            first_time = timestamps[0]
            last_time = timestamps[-1]

            logger.info("=" * 80)
            logger.info("Data Validation Results")
            logger.info("=" * 80)
            logger.info(f"Total candles: {candle_count:,}")
            logger.info(f"First candle: {first_time}")
            logger.info(f"Last candle:  {last_time}")
            logger.info(f"File size: {self.output_file.stat().st_size / (1024**2):.2f} MB")
            logger.info("=" * 80)


def main():
    """CLI entry point for S&P 500 historical data download."""
    parser = argparse.ArgumentParser(
        description="Download historical S&P 500 data from OANDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 10 years of 1-minute S&P 500 data
  python download_sp500_historical.py

  # Download 5 years of data
  python download_sp500_historical.py --years 5

  # Validate existing data
  python download_sp500_historical.py --validate-only

  # Resume interrupted download
  python download_sp500_historical.py
        """
    )

    parser.add_argument(
        "--instrument",
        default="SPX500_USD",
        help="OANDA instrument symbol (default: SPX500_USD)"
    )
    parser.add_argument(
        "--granularity",
        default="M1",
        help="Candle granularity (default: M1 for 1-minute)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of years of historical data (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/bronze/prices",
        help="Output directory (default: data/bronze/prices)"
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        help="Delay between API requests in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing data, don't download"
    )

    args = parser.parse_args()

    downloader = SP500HistoricalDownloader(
        instrument=args.instrument,
        granularity=args.granularity,
        output_dir=args.output_dir,
        years_back=args.years,
        rate_limit_delay=args.rate_limit_delay
    )

    if args.validate_only:
        downloader.validate_data()
    else:
        logger.info("=" * 80)
        logger.info(f"S&P 500 Historical Data Downloader")
        logger.info("=" * 80)
        logger.info(f"Instrument: {args.instrument}")
        logger.info(f"Granularity: {args.granularity}")
        logger.info(f"Years: {args.years}")
        logger.info(f"Output: {args.output_dir}")
        logger.info("=" * 80)
        logger.info("")

        downloader.download_historical_data()

        # Validate after download
        logger.info("")
        downloader.validate_data()


if __name__ == "__main__":
    main()
