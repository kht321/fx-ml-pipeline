"""
Join news manifest metadata with quantitative gold features on timestamp.

This utility performs a left join of the news manifest table against a
quantitative gold table (e.g., market signals) so that each news article
is enriched with the corresponding numeric features.

Example usage:

    python -m src_clean.data_pipelines.gold.news_manifest_join \
        --manifest-path data_clean/silver/news/news_manifest_clean.parquet \
        --gold-path data_clean/gold/market/features/spx500_features.parquet \
        --output-path data_clean/gold/news/news_with_features.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def read_table(path: Path) -> pd.DataFrame:
    """Read parquet or CSV into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")

    ext = path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension '{ext}' for {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Persist DataFrame to parquet or CSV based on extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif ext in {".csv", ".txt"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension '{ext}' for {path}")


def coerce_datetime(series: pd.Series) -> pd.Series:
    """Convert a series to datetime with UTC if possible."""
    converted = pd.to_datetime(series, errors="coerce", utc=True)
    if converted.isna().all():
        raise ValueError("Timestamp column contains no parsable datetime values.")
    return converted


def join_manifest_with_gold(
    manifest_path: Path,
    gold_path: Path,
    output_path: Path,
    manifest_ts_col: str = "published_at",
    gold_ts_col: str = "timestamp",
) -> Path:
    """Left join manifest metadata with gold features on matching timestamps."""
    manifest_df = read_table(manifest_path)
    gold_df = read_table(gold_path)

    if manifest_ts_col not in manifest_df.columns:
        raise KeyError(f"Manifest missing timestamp column '{manifest_ts_col}'.")
    if gold_ts_col not in gold_df.columns:
        raise KeyError(f"Gold table missing timestamp column '{gold_ts_col}'.")

    manifest_df = manifest_df.copy()
    gold_df = gold_df.copy()

    manifest_df[manifest_ts_col] = coerce_datetime(manifest_df[manifest_ts_col])
    gold_df[gold_ts_col] = coerce_datetime(gold_df[gold_ts_col])

    logger.info(
        "Joining manifest (%d rows) with gold table (%d rows) on %s == %s",
        len(manifest_df),
        len(gold_df),
        manifest_ts_col,
        gold_ts_col,
    )

    merged = manifest_df.merge(
        gold_df,
        how="left",
        left_on=manifest_ts_col,
        right_on=gold_ts_col,
        suffixes=("", "_gold"),
    )

    write_table(merged, output_path)
    logger.info("Enriched news table saved to %s (%d rows).", output_path, len(merged))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join news manifest with gold features.")
    parser.add_argument("--manifest-path", type=Path, required=True, help="Path to manifest parquet/csv.")
    parser.add_argument("--gold-path", type=Path, required=True, help="Path to gold features parquet/csv.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination parquet/csv.")
    parser.add_argument(
        "--manifest-ts-col",
        type=str,
        default="published_at",
        help="Timestamp column in manifest (default: published_at).",
    )
    parser.add_argument(
        "--gold-ts-col",
        type=str,
        default="timestamp",
        help="Timestamp column in gold table (default: timestamp).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    join_manifest_with_gold(
        manifest_path=args.manifest_path,
        gold_path=args.gold_path,
        output_path=args.output_path,
        manifest_ts_col=args.manifest_ts_col,
        gold_ts_col=args.gold_ts_col,
    )


if __name__ == "__main__":
    main()
