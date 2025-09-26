"""Fetch the order book snapshot and print the JSON payload."""

import json
import sys

from oanda_api import fetch_orderbook


def main() -> None:
    instrument = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"

    payload = fetch_orderbook(instrument)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
