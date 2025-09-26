"""Stream live prices from OANDA and print tick JSON to stdout."""

import json
import os
import sys

from oanda_api import stream_prices


def main() -> None:
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    if not account_id:
        sys.exit("Missing OANDA_ACCOUNT_ID in environment")

    instruments = sys.argv[1:] or ["EUR_USD", "GBP_USD"]

    for msg_type, message in stream_prices(account_id, instruments):
        if msg_type == "price":
            print(json.dumps(message), flush=True)


if __name__ == "__main__":
    main()
