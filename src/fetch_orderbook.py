"""Fetch the order book snapshot and print the JSON payload.

Only a subset of major currency pairs expose order-book snapshots. This helper
wraps the API call and adds user-friendly error handling so unsupported pairs
produce a clear message rather than a traceback.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from oandapyV20.exceptions import V20Error

from oanda_api import fetch_orderbook


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Describe CLI options for choosing the instrument and sink."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instrument", nargs="?", default="EUR_USD")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write payload to this path instead of stdout",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    """Run the order-book request and persist the resulting JSON."""
    args = parse_args(argv or sys.argv[1:])

    try:
        payload = fetch_orderbook(args.instrument)
    except V20Error as exc:  # Surface instrument eligibility or auth issues
        sys.stderr.write(f"Failed to fetch order book for {args.instrument}: {exc}\n")
        sys.exit(1)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload), encoding="utf-8")
    else:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
