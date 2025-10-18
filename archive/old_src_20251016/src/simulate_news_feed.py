"""Drip-feed curated news into the Bronze layer and trigger the prediction loop.

Run this helper when you want to simulate the irregular arrival of news
headlines. It copies articles from a corpus directory into the live Bronze news
folder, invokes the existing ingestion/feature-building scripts, and optionally
rebuilds the Gold training set and baseline model so the latest prediction is
refreshed for front-end consumption.
"""

import argparse
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/bronze/news_corpus"),
        help="Directory containing seed news stories to drip into the live feed",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/bronze/news"),
        help="Bronze news directory monitored by process_news.py",
    )
    parser.add_argument(
        "--silver-path",
        type=Path,
        default=Path("data/silver/news/news_features.csv"),
        help="Destination CSV for engineered news features",
    )
    parser.add_argument(
        "--price-features",
        type=Path,
        default=Path("data/silver/prices/sgd_vs_majors.csv"),
        help="Silver price feature table used to rebuild the Gold dataset",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=Path("data/gold/training/sgd_vs_majors_training.csv"),
        help="Gold training dataset to regenerate after each drop",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Base sleep in seconds between article drops",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=10.0,
        help="Maximum random jitter (seconds) added to each interval",
    )
    parser.add_argument(
        "--rebuild-gold",
        action="store_true",
        help="Re-run build_training_set.py after each ingest",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Re-run train_baseline.py after each ingest to refresh predictions",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("data/gold/models/logreg_latest.pkl"),
        help="Path where retrained models should be stored",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them",
    )
    return parser.parse_args()


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print(f"[simulate_news_feed] executing: {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"Corpus directory not found: {args.corpus}")

    args.target.mkdir(parents=True, exist_ok=True)

    corpus_files = sorted(
        [p for p in args.corpus.glob("*") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )

    if not corpus_files:
        print("[simulate_news_feed] no corpus files to stream")
        return

    process_cmd = [
        sys.executable,
        "src/process_news.py",
        "--input-dir",
        str(args.target),
        "--silver-path",
        str(args.silver_path),
    ]

    gold_cmd = [
        sys.executable,
        "src/build_training_set.py",
        "--price-features",
        str(args.price_features),
        "--news-features",
        str(args.silver_path),
        "--output",
        str(args.gold_path),
    ]

    train_cmd = [
        sys.executable,
        "src/train_baseline.py",
        str(args.gold_path),
        "--model-output",
        str(args.model_output),
    ]

    for idx, file_path in enumerate(corpus_files, start=1):
        target_path = args.target / file_path.name
        if target_path.exists():
            print(f"[simulate_news_feed] skipping existing file: {target_path}")
            continue

        shutil.copy2(file_path, target_path)
        print(f"[simulate_news_feed] dropped {target_path.name} ({idx}/{len(corpus_files)})")

        run_command(process_cmd, dry_run=args.dry_run)

        if args.rebuild_gold:
            run_command(gold_cmd, dry_run=args.dry_run)

        if args.retrain:
            run_command(train_cmd, dry_run=args.dry_run)

        if idx < len(corpus_files):
            base = max(args.interval, 0.0)
            jitter = random.uniform(0, max(args.jitter, 0.0))
            sleep_for = base + jitter
            print(f"[simulate_news_feed] sleeping {sleep_for:.1f}s before next drop")
            if not args.dry_run:
                time.sleep(sleep_for)

    print("[simulate_news_feed] feed completed")


if __name__ == "__main__":
    main()
