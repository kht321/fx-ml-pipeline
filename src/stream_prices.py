"""Stream live prices from OANDA and tee tick JSON into Bronze/Silver layers."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

from oanda_api import stream_prices


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instruments",
        nargs="*",
        default=["USD_SGD", "EUR_USD", "GBP_USD"],
        help="Instrument symbols to stream (default: USD_SGD EUR_USD GBP_USD)",
    )
    parser.add_argument(
        "--account-id",
        dest="account_id",
        default=os.getenv("OANDA_ACCOUNT_ID"),
        help="Override OANDA account id (env OANDA_ACCOUNT_ID)",
    )
    parser.add_argument(
        "--bronze-path",
        type=Path,
        help="Optional path to append raw price ticks (Bronze layer)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Stop after streaming N price ticks",
    )
    parser.add_argument(
        "--suppress-stdout",
        action="store_true",
        help="Do not emit ticks to stdout (useful when only recording Bronze)",
    )
    parser.add_argument(
        "--include-heartbeats",
        action="store_true",
        help="Persist heartbeat messages alongside price ticks in Bronze output",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    if not args.account_id:
        sys.exit("Missing OANDA_ACCOUNT_ID in environment or --account-id")

    bronze_handle = None
    if args.bronze_path:
        args.bronze_path.parent.mkdir(parents=True, exist_ok=True)
        bronze_handle = args.bronze_path.open("a", encoding="utf-8")

    tick_count = 0

    try:
        for msg_type, message in stream_prices(args.account_id, args.instruments):
            is_price = msg_type == "price"
            should_persist = is_price or args.include_heartbeats
            if bronze_handle and should_persist:
                bronze_handle.write(json.dumps(message) + "\n")
                bronze_handle.flush()

            if is_price and not args.suppress_stdout:
                print(json.dumps(message), flush=True)

            if is_price:
                tick_count += 1
                if args.max_ticks is not None and tick_count >= args.max_ticks:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if bronze_handle:
            bronze_handle.close()


if __name__ == "__main__":
    main()
