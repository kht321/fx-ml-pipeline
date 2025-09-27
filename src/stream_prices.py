"""Stream live FX prices from OANDA into the medallion pipeline.

This script is the entry point for the Bronze layer. It maintains a streaming
connection to the OANDA pricing endpoint, optionally writes every message to a
newline-delimited file for archival purposes, and forwards price ticks to
stdout so downstream feature builders can consume them in near-real time.
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

from oandapyV20.exceptions import V20Error

from oanda_api import stream_prices


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Describe and parse the CLI interface for the streamer."""
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
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N price ticks (0 to disable)",
    )
    return parser.parse_args(list(argv))


def log(message: str) -> None:
    """Emit structured progress messages to stderr for easier monitoring."""
    sys.stderr.write(f"[stream_prices] {message}\n")
    sys.stderr.flush()


def main(argv: Iterable[str] | None = None) -> None:
    """Start the streaming loop and coordinate persistence/forwarding."""
    args = parse_args(argv or sys.argv[1:])

    if not args.account_id:
        sys.exit("Missing OANDA_ACCOUNT_ID in environment or --account-id")

    if args.bronze_path:
        args.bronze_path.parent.mkdir(parents=True, exist_ok=True)

    bronze_ctx = (
        args.bronze_path.open("a", encoding="utf-8")
        if args.bronze_path
        else nullcontext()
    )

    tick_count = 0
    heartbeat_count = 0

    try:
        with bronze_ctx as bronze_handle:
            for msg_type, message in stream_prices(args.account_id, args.instruments):
                is_price = msg_type == "price"
                is_heartbeat = msg_type == "heartbeat"

                if bronze_handle and (is_price or (is_heartbeat and args.include_heartbeats)):
                    bronze_handle.write(json.dumps(message) + "\n")
                    bronze_handle.flush()

                if is_price:
                    tick_count += 1
                    if not args.suppress_stdout:
                        print(json.dumps(message), flush=True)

                    if args.log_every and tick_count % args.log_every == 0:
                        log(f"streamed {tick_count} price ticks")

                    if args.max_ticks is not None and tick_count >= args.max_ticks:
                        break
                elif is_heartbeat:
                    heartbeat_count += 1
    except KeyboardInterrupt:
        log("interrupted by user; closing stream")
    except V20Error as exc:
        log(f"API error: {exc}")
        sys.exit(1)
    finally:
        log(
            f"finished with {tick_count} price ticks and {heartbeat_count} heartbeats"
        )


if __name__ == "__main__":
    main()
