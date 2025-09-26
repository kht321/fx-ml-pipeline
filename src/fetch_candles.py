"""Fetch historical candles from OANDA and print the JSON payload."""

import json
import sys

from oanda_api import fetch_candles


def main() -> None:
    instrument = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"
    granularity = sys.argv[2] if len(sys.argv) > 2 else "M1"
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    payload = fetch_candles(instrument, granularity, count)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
