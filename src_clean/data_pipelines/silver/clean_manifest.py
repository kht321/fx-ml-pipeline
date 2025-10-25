"""
Utilities to clean and refresh the news manifest produced from bronze JSON files.

Steps performed:
    1. Remove duplicate headlines, keeping the earliest published timestamp.
    2. Drop rows whose underlying article body is empty or missing.

Usage:
    python -m src_clean.data_pipelines.silver.clean_manifest \
        --manifest-path data_clean/silver/news/news_manifest.parquet \
        --output-path data_clean/silver/news/news_manifest_clean.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_parquet(manifest_path)


def _warn_dropped(before: int, after: int, message: str) -> None:
    dropped = before - after
    if dropped > 0:
        logger.warning("%s Dropped %d row(s).", message, dropped)
    else:
        logger.info("%s No rows dropped.", message)


def _timestamped_path(path: Path) -> Path:
    """Append a UTC timestamp to the filename before the suffix."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = path.suffix or ".parquet"
    new_name = f"{path.stem}_cleaned_{timestamp}{suffix}"
    return path.with_name(new_name)


def _load_article_body(file_path: str) -> str:
    path = Path(file_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("body") or "").strip()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to read %s: %s", path, exc)
        return ""


def clean_manifest_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply duplicate/empty-body cleaning rules to the manifest DataFrame."""
    df = df.sort_values("published_at", ascending=True)
    

    before = len(df)
    df = df.dropna(subset=["headline"])
    df = df.drop_duplicates(subset=["headline"], keep="first")
    _warn_dropped(before, len(df), "Removed duplicate headlines.")

    before = len(df)
    body_mask = df["content_fetched"]
    df = df[body_mask]
    _warn_dropped(before, len(df), "Removed rows with empty article bodies.")

    return df.reset_index(drop=True)


def clean_manifest(manifest_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Load a manifest, run cleaning steps, and save the cleaned parquet.

    Parameters
    ----------
    manifest_path:
        Existing manifest parquet path.
    output_path:
        Destination for the cleaned manifest. If omitted, a timestamped file
        (e.g. `manifest_20250221_173045.parquet`) is created alongside the input.
    """
    df = _load_manifest(manifest_path)
    print(df.columns)
    cleaned = clean_manifest_df(df)

    target = output_path or _timestamped_path(manifest_path)
    if output_path is None:
        logger.info("No output path supplied; using timestamped file %s.", target)

    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(target, index=False)
    logger.info("Cleaned manifest saved to %s (%d rows).", target, len(cleaned))
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a news manifest parquet.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to the existing manifest parquet.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for cleaned manifest (defaults to timestamped filename).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    clean_manifest(args.manifest_path, args.output_path)


if __name__ == "__main__":
    main()
