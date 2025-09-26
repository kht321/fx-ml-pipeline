"""Build simple features from streamed OANDA price ticks."""

import json
import sys
from typing import Iterable

import numpy as np
import pandas as pd


def to_price_frame(messages: Iterable[str]) -> pd.DataFrame:
    records = [json.loads(line) for line in messages if line.strip()]
    return pd.DataFrame.from_records(records)


def main() -> None:
    df = to_price_frame(sys.stdin)
    if df.empty:
        sys.exit("No price data on stdin")

    bids = df["bids"].str[0].str["price"].astype(float)
    asks = df["asks"].str[0].str["price"].astype(float)
    mid = (bids + asks) / 2
    mid.name = "mid"

    features = pd.DataFrame(
        {
            "ret_1": mid.pct_change(),
            "roll_mean_5": mid.rolling(5).mean().pct_change(),
            "roll_vol_20": mid.pct_change().rolling(20).std(),
        }
    ).dropna()

    target = (mid.shift(-5) > mid).loc[features.index].astype(int)
    dataset = features.join(target.rename("y"))

    dataset.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
