"""Promote Silver price & news features into a Gold training table.

The Gold layer needs a single, analysis-ready table that combines market
features with contextual sentiment signals. This utility merges the two Silver
feeds using an as-of join so each price observation carries the most recent
relevant news headline within a configurable lookback window.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Set

import numpy as np
import pandas as pd


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Build the argument parser so callers can configure inputs/outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--price-features",
        type=Path,
        default=Path("data/silver/prices/sgd_vs_majors.csv"),
        help="CSV containing engineered price features",
    )
    parser.add_argument(
        "--news-features",
        type=Path,
        default=Path("data/silver/news/news_features.csv"),
        help="CSV of engineered news features",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gold/training/sgd_vs_majors_training.csv"),
        help="Destination CSV for the Gold training table",
    )
    parser.add_argument(
        "--news-tolerance",
        default="6H",
        help="Max lookback window for joining news to price features (e.g. 6H, 2H)",
    )
    return parser.parse_args(list(argv))


def parse_currency_mentions(value: str) -> Set[str]:
    """Split the comma-delimited currency codes into a comparable set."""
    if not value or pd.isna(value):
        return set()
    return {token.strip().upper() for token in str(value).split(",") if token.strip()}


def prepare_news_frame(path: Path) -> pd.DataFrame:
    """Load engineered news features and precompute helper columns."""
    if not path.exists():
        return pd.DataFrame()

    news_df = pd.read_csv(path)
    if news_df.empty:
        return news_df

    news_df["published_at"] = pd.to_datetime(news_df["published_at"], utc=True, errors="coerce")
    news_df = news_df.dropna(subset=["published_at"])
    news_df["currency_set"] = news_df["currency_mentions"].apply(parse_currency_mentions)
    news_df.sort_values("published_at", inplace=True)
    return news_df


def filter_instrument_news(news_df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Restrict the news feed to headlines that mention the instrument legs."""
    if news_df.empty:
        return news_df

    base, quote = instrument.split("_", maxsplit=1)
    mask = news_df["currency_set"].apply(
        lambda codes: bool(codes.intersection({base, quote, "SGD"}))
    )
    return news_df.loc[mask].copy()


def join_news(price_df: pd.DataFrame, news_df: pd.DataFrame, tolerance: str) -> pd.DataFrame:
    """Perform an as-of merge matching each price row to the latest headline."""
    if price_df.empty:
        return price_df

    price_df = price_df.copy()
    price_df["time"] = pd.to_datetime(price_df["time"], utc=True, errors="coerce")
    price_df = price_df.dropna(subset=["time", "instrument", "y"])  # y already computed upstream

    if news_df.empty:
        price_df["news_sentiment_score"] = 0.0
        price_df["news_mentions_sgd"] = 0
        price_df["news_word_count"] = 0
        price_df["news_age_minutes"] = np.nan
        price_df["news_story_id"] = pd.NA
        price_df["news_headline"] = pd.NA
        price_df["news_source"] = pd.NA
        return price_df

    merged_frames: List[pd.DataFrame] = []
    td = pd.to_timedelta(tolerance)

    # Process each instrument separately to guarantee we only merge against the
    # subset of headlines that talk about its component currencies.
    for instrument, group in price_df.groupby("instrument"):
        relevant_news = filter_instrument_news(news_df, instrument)
        if relevant_news.empty:
            temp = group.copy()
            temp["news_sentiment_score"] = 0.0
            temp["news_mentions_sgd"] = 0
            temp["news_word_count"] = 0
            temp["news_age_minutes"] = np.nan
            temp["news_story_id"] = pd.NA
            temp["news_headline"] = pd.NA
            temp["news_source"] = pd.NA
            merged_frames.append(temp)
            continue

        join_cols = [
            "published_at",
            "sentiment_score",
            "mentions_sgd",
            "word_count",
            "story_id",
            "headline",
            "source",
        ]
        relevant_news = relevant_news.loc[:, join_cols].sort_values("published_at")

        merged = pd.merge_asof(
            group.sort_values("time"),
            relevant_news,
            left_on="time",
            right_on="published_at",
            direction="backward",
            tolerance=td,
        )

        merged["news_sentiment_score"] = merged["sentiment_score"].fillna(0.0)
        merged["news_mentions_sgd"] = merged["mentions_sgd"].fillna(0).astype(int)
        merged["news_word_count"] = merged["word_count"].fillna(0)
        merged["news_story_id"] = merged["story_id"]
        merged["news_headline"] = merged["headline"]
        merged["news_source"] = merged["source"]
        merged["news_age_minutes"] = (
            (merged["time"] - merged["published_at"]).dt.total_seconds() / 60
        )

        merged_frames.append(merged.drop(columns=[
            "sentiment_score",
            "mentions_sgd",
            "word_count",
            "story_id",
            "headline",
            "source",
        ]))

    combined = pd.concat(merged_frames, ignore_index=True)
    combined.sort_values(["instrument", "time"], inplace=True)
    return combined


def main(argv: Iterable[str] | None = None) -> None:
    """Wire parsing, loading, joining, and final persistence together."""
    args = parse_args(argv or sys.argv[1:])

    if not args.price_features.exists():
        raise SystemExit(f"Price feature file not found: {args.price_features}")

    price_df = pd.read_csv(args.price_features)
    news_df = prepare_news_frame(args.news_features)

    combined = join_news(price_df, news_df, args.news_tolerance)
    combined["time"] = combined["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if "published_at" in combined:
        combined["published_at"] = combined["published_at"].dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
