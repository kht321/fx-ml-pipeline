"""Build multi-horizon label store for FX instruments.

Outputs labels at (instrument, event_timestamp) for horizons:
- 15m, 60m returns & direction
- 6h, 24h returns
- 24h rolling volatility
- 24h high-low spread
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices", type=Path, default=Path("data/market/silver/technical_features/sgd_vs_majors.csv"), help="Price features table with at least instrument, time, close")
    p.add_argument("--output", type=Path, default=Path("data/combined/gold/labels/labels.parquet"))
    return p.parse_args(list(argv) if argv is not None else None)


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: instrument, time (or timestamp), close, high, low
    if "time" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    df["event_timestamp"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.sort_values(["instrument", "event_timestamp"]).copy()

    out_frames = []
    for inst, g in df.groupby("instrument"):
        g = g.copy()
        g["close"] = pd.to_numeric(g.get("close", g.get("price", g.iloc[:, -1])), errors="coerce")

        # Returns over horizons
        for mins in [15, 60]:
            f = g["close"].shift(-(mins))  # assuming 1-minute granularity; if hourly, adjust
            ret = (f - g["close"]) / g["close"]
            g[f"return_{mins}min"] = ret
            g[f"direction_{mins}min"] = (ret > 0).astype("Int64")

        for hours in [6, 24]:
            steps = hours * 60
            f = g["close"].shift(-(steps))
            g[f"return_{hours}hour"] = (f - g["close"]) / g["close"]

        # 24h rolling volatility (std of minute returns)
        ret1 = g["close"].pct_change()
        g["volatility_24hour"] = ret1.rolling(24 * 60, min_periods=60).std()

        # 24h high-low spread
        if "high" in g.columns and "low" in g.columns and "open" in g.columns:
            # rolling normalized range over last 24h
            high_roll = g["high"].rolling(24 * 60, min_periods=60).max()
            low_roll = g["low"].rolling(24 * 60, min_periods=60).min()
            open_roll = g["open"].rolling(24 * 60, min_periods=60).mean()
            g["high_low_spread_24hour"] = (high_roll - low_roll) / open_roll
        else:
            g["high_low_spread_24hour"] = pd.NA

        out_frames.append(g[[
            "instrument", "event_timestamp",
            "return_15min", "direction_15min",
            "return_60min", "direction_60min",
            "return_6hour", "return_24hour",
            "volatility_24hour", "high_low_spread_24hour",
        ]])

    labels = pd.concat(out_frames, ignore_index=True)
    return labels.dropna(subset=["event_timestamp"])  # keep rows with valid timestamps


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if not args.prices.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["instrument", "event_timestamp"]).to_parquet(args.output, index=False)
        return
    if args.prices.suffix.lower() == ".csv":
        prices = pd.read_csv(args.prices)
    else:
        prices = pd.read_parquet(args.prices)

    labels = build_labels(prices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()

