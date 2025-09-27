"""Transform Bronze tick data into Silver-level engineered features.

This module reads streaming tick JSON (either from stdin or an NDJSON file),
computes per-instrument technical features, and writes the results incrementally
so Silver datasets can be refreshed while the stream is still running.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Define the command-line interface and parse user inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to newline-delimited tick JSON (defaults to stdin)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV destination for Silver features (defaults to stdout)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Tick horizon for the binary target label",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=25,
        help="Require at least N price rows before emitting features",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=100,
        help="Flush engineered rows every N price ticks",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log progress every N price ticks",
    )
    return parser.parse_args(list(argv))


def log(message: str) -> None:
    """Write progress information to stderr for operational visibility."""
    sys.stderr.write(f"[build_features] {message}\n")
    sys.stderr.flush()


def to_price_frame(records: List[dict]) -> pd.DataFrame:
    """Convert buffered JSON records into a DataFrame expected by pandas ops."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def engineer_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create rolling statistical features and the classification label."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "instrument", "bids", "asks"]).reset_index(drop=True)

    feature_frames = []
    for instrument, group in df.groupby("instrument"):
        group = group.sort_values("time")
        if len(group) < 5:
            continue

        bids = group["bids"].str[0].str["price"].astype(float)
        asks = group["asks"].str[0].str["price"].astype(float)
        bid_liq = group["bids"].str[0].str.get("liquidity").astype(float)
        ask_liq = group["asks"].str[0].str.get("liquidity").astype(float)

        mid = (bids + asks) / 2
        spread = asks - bids
        ret_1 = mid.pct_change()

        features = pd.DataFrame(
            {
                "time": group["time"],
                "instrument": instrument,
                "mid": mid,
                "spread": spread,
                "ret_1": ret_1,
                "ret_5": mid.pct_change(periods=5),
                "roll_vol_20": ret_1.rolling(20).std(),
                "zscore_20": (mid - mid.rolling(20).mean()) / mid.rolling(20).std(),
                "bid_liquidity": bid_liq,
                "ask_liquidity": ask_liq,
            }
        )

        features["y"] = (mid.shift(-horizon) > mid).astype(float)
        features = features.dropna()
        feature_frames.append(features)

    if not feature_frames:
        return pd.DataFrame()

    dataset = pd.concat(feature_frames, ignore_index=True)
    dataset.sort_values(["instrument", "time"], inplace=True)
    return dataset.reset_index(drop=True)


def flush_features(
    records: List[dict],
    horizon: int,
    output_path: Path | None,
    headers_written: bool,
    rows_written: int,
) -> tuple[int, bool]:
    """Persist newly generated rows without re-writing the entire dataset."""
    df = to_price_frame(records)
    dataset = engineer_features(df, horizon)

    if dataset.empty or len(dataset) <= rows_written:
        return rows_written, headers_written

    new_rows = dataset.iloc[rows_written:]
    new_rows["time"] = new_rows["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_rows.to_csv(
            output_path,
            mode="a" if headers_written else "w",
            header=not headers_written,
            index=False,
        )
        headers_written = True
    else:
        new_rows.to_csv(sys.stdout, header=not headers_written, index=False)
        sys.stdout.flush()
        headers_written = True

    rows_written = len(dataset)
    log(
        f"wrote {len(new_rows)} new feature rows (total {rows_written})"
    )
    return rows_written, headers_written


def main(argv: Iterable[str] | None = None) -> None:
    """Stream ticks from stdin or a file, engineering features as we go."""
    args = parse_args(argv or sys.argv[1:])

    records: List[dict] = []
    total_lines = 0
    price_count = 0
    headers_written = False
    rows_written = 0

    # ``handle_line`` normalises each incoming line and stores price ticks so
    # that the incremental flush logic can detect when enough data has accrued
    # to justify recomputing the feature window.
    def handle_line(line: str) -> None:
        nonlocal total_lines, price_count
        line = line.strip()
        if not line:
            return
        total_lines += 1
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            log(f"skipping invalid JSON line: {line[:60]}...")
            return

        msg_type = str(message.get("type", "price")).lower()
        if msg_type != "price":
            return

        records.append(message)
        price_count += 1

        if args.log_every and price_count % args.log_every == 0:
            log(f"processed {price_count} price ticks")

    try:
        if args.input:
            for raw_line in args.input.read_text(encoding="utf-8").splitlines():
                handle_line(raw_line)
            rows_written, headers_written = flush_features(
                records,
                args.horizon,
                args.output,
                headers_written,
                rows_written,
            )
        else:
            for raw_line in sys.stdin:
                handle_line(raw_line)

                if (
                    price_count >= args.min_rows
                    and args.flush_interval
                    and price_count % args.flush_interval == 0
                ):
                    rows_written, headers_written = flush_features(
                        records,
                        args.horizon,
                        args.output,
                        headers_written,
                        rows_written,
                    )

            rows_written, headers_written = flush_features(
                records,
                args.horizon,
                args.output,
                headers_written,
                rows_written,
            )
    except KeyboardInterrupt:
        log("interrupted; flushing remaining rows")
        rows_written, headers_written = flush_features(
            records,
            args.horizon,
            args.output,
            headers_written,
            rows_written,
        )


    log(f'completed run: {price_count} price ticks, {rows_written} feature rows written')
    if price_count < args.min_rows:
        log(
            f"only {price_count} price ticks received; minimum required is {args.min_rows}"
        )


if __name__ == "__main__":
    main()
