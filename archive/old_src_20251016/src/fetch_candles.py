"""Fetch historical candles from OANDA and persist them to the Bronze layer.

This CLI wrapper exists so analysts can snapshot the market state on demand
without writing API boilerplate each time. It emits the raw JSON payload OANDA
returns so downstream transforms have the full fidelity data available.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from oanda_api import fetch_candles


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Declare CLI options for instrument selection and persistence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instrument", help="Instrument to request (e.g. USD_SGD)")
    parser.add_argument(
        "--granularity",
        default="M1",
        help="Candle granularity (default: M1)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of candles to request (default: 500)",
    )
    parser.add_argument(
        "--price",
        default="MBA",
        help="Price types to include (default: MBA)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write payload to this path instead of stdout",
    )
    return parser.parse_args(list(argv))


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Execute the REST call and fan the payload out to stdout or a file."""
    args = parse_args(argv or sys.argv[1:])
    payload = fetch_candles(
        instrument=args.instrument,
        granularity=args.granularity,
        count=args.count,
        price=args.price,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload), encoding="utf-8")
    else:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
