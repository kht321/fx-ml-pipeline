"""
Utilities for building and querying a manifest of bronze news articles.

The manifest is stored as a Parquet file and contains lightweight metadata
about every JSON article so downstream code can filter quickly without
touching the entire filesystem.

Example CLI usage:

    python -m src_clean.data_pipelines.silver.news_manifest \
        --input-dir data_clean/bronze/news/historical_5year \
        --output-path data_clean/silver/news/news_manifest.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _iter_article_metadata(input_dir: Path, pattern: str = "*.json") -> Iterable[dict]:
    """Yield minimal metadata for each JSON article file."""
    for path in sorted(input_dir.glob(pattern)):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to parse %s: %s", path, exc)
            continue

        yield {
            "article_id": data.get("article_id"),
            "headline": data.get("headline"),
            "source": data.get("source"),
            "published_at": data.get("published_at"),
            "collected_at": data.get("collected_at"),
            "language": data.get("language"),
            "sp500_relevant": data.get("sp500_relevant"),
            "file_path": str(path),
            "content_fetched": data.get("content_fetched"), 
            "file_size_bytes": path.stat().st_size,
        }


def build_manifest(input_dir: Path, output_path: Path, pattern: str = "*.json") -> Path:
    """
    Generate a manifest parquet containing lightweight metadata.

    Parameters
    ----------
    input_dir:
        Directory containing JSON news files.
    output_path:
        Target parquet file.
    pattern:
        Glob pattern to match files (default: *.json).
    """
    rows: List[dict] = list(_iter_article_metadata(input_dir, pattern))
    if not rows:
        raise FileNotFoundError(f"No JSON files found in {input_dir} matching {pattern}")

    df = pd.DataFrame(rows)

    # Parse timestamps to datetime for easier filtering later.
    for col in ["published_at", "collected_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df = df.sort_values("published_at", ascending=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)
    return output_path


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load the manifest parquet as a DataFrame."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_parquet(manifest_path)


def filter_manifest(
    manifest_path: Path,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
    sources: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Filter the manifest by published date range and optional sources.

    Parameters
    ----------
    manifest_path:
        Path to the parquet file.
    start, end:
        Inclusive datetime bounds. If omitted, no bound is applied.
    limit:
        Maximum number of rows to return (after sorting by published_at).
    sources:
        Optional iterable of sources to include.
    """
    df = load_manifest(manifest_path)
    if start is not None:
        df = df[df["published_at"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        df = df[df["published_at"] <= pd.to_datetime(end, utc=True)]
    if sources:
        df = df[df["source"].isin(set(sources))]

    df = df.sort_values("published_at", ascending=True)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a news manifest parquet.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of JSON articles.")
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output parquet path (e.g. data_clean/silver/news/news_manifest.parquet).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for article files (default: *.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_manifest(args.input_dir, args.output_path, args.pattern)


if __name__ == "__main__":
    main()
