"""Fetch the order book snapshot and print the JSON payload."""

import json
import sys

from oandapyV20.exceptions import V20Error

from oanda_api import fetch_orderbook


def main() -> None:
    instrument = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"

    try:
        payload = fetch_orderbook(instrument)
    except V20Error as exc:  # Surface instrument eligibility or auth issues
        sys.stderr.write(f"Failed to fetch order book for {instrument}: {exc}\n")
        sys.exit(1)

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
