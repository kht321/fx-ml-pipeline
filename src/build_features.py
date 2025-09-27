"""Transform Bronze tick data into Silver-level engineered features."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
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
    return parser.parse_args(list(argv))


def read_messages(args: argparse.Namespace) -> List[str]:
    if args.input:
        return args.input.read_text(encoding="utf-8").splitlines()
    return [line for line in sys.stdin]


def to_price_frame(messages: Iterable[str]) -> pd.DataFrame:
    records = []
    for line in messages:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame.from_records(records)


def engineer_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "instrument"])

    feature_frames = []
    for instrument, group in df.groupby("instrument"):
        group = group.sort_values("time")
        if len(group) < 5:
            continue

        bids = group["bids"].str[0].str["price"].astype(float)
        asks = group["asks"].str[0].str["price"].astype(float)
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
                "bid_liquidity": group["bids"].str[0].str.get("liquidity").astype(float),
                "ask_liquidity": group["asks"].str[0].str.get("liquidity").astype(float),
            }
        )

        features["y"] = (mid.shift(-horizon) > mid).astype(float)
        features = features.dropna()
        feature_frames.append(features)

    if not feature_frames:
        return pd.DataFrame()

    return pd.concat(feature_frames, ignore_index=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    messages = read_messages(args)
    df = to_price_frame(messages)

    if len(df) < args.min_rows:
        sys.exit("Insufficient price data for feature engineering")

    dataset = engineer_features(df, args.horizon)
    if dataset.empty:
        sys.exit("No engineered features produced")

    dataset.sort_values(["instrument", "time"], inplace=True)
    dataset["time"] = dataset["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(args.output, index=False)
    else:
        dataset.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
