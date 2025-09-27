"""Convert curated news drops into Silver-level textual features."""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

POSITIVE_LEXICON = {
    "growth",
    "gain",
    "improve",
    "strong",
    "bullish",
    "increase",
    "optimistic",
    "upgrade",
}
NEGATIVE_LEXICON = {
    "fall",
    "risk",
    "slowdown",
    "bearish",
    "decline",
    "downgrade",
    "weak",
    "loss",
}
CURRENCY_CODES = {
    "usd",
    "sgd",
    "eur",
    "gbp",
    "jpy",
    "aud",
    "chf",
    "cny",
    "hkd",
}
SGD_KEY_TERMS = {"singapore", "mas", "lion city", "sgd"}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/bronze/news"),
        help="Directory containing curated news documents",
    )
    parser.add_argument(
        "--silver-path",
        type=Path,
        default=Path("data/silver/news/news_features.csv"),
        help="Destination CSV for engineered news features",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/bronze/news/.processed.json"),
        help="Manifest that tracks processed files to avoid reprocessing",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep polling the directory for newly dropped files",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between polls when --follow is enabled",
    )
    return parser.parse_args(list(argv))


def load_manifest(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except json.JSONDecodeError:
        return set()


def save_manifest(path: Path, processed: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(processed), indent=2))


def read_news(file_path: Path) -> Dict[str, str]:
    if file_path.suffix.lower() == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        text = data.get("body") or data.get("text") or ""
        headline = data.get("headline") or data.get("title") or file_path.stem
        published_at = data.get("published_at") or data.get("time")
        source = data.get("source", "unknown")
    else:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        headline = file_path.stem
        published_at = None
        source = "curated"

    if not published_at:
        published_at = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(file_path.stat().st_mtime)
        )

    return {
        "story_id": file_path.stem,
        "headline": headline,
        "text": text,
        "published_at": published_at,
        "source": source,
        "path": str(file_path.resolve()),
    }


def extract_features(news: Dict[str, str]) -> Dict[str, object]:
    tokens = re.findall(r"[a-zA-Z']+", news["text"].lower())
    token_count = len(tokens)
    unique_tokens = len(set(tokens))

    pos_hits = sum(1 for token in tokens if token in POSITIVE_LEXICON)
    neg_hits = sum(1 for token in tokens if token in NEGATIVE_LEXICON)
    sentiment_score = 0.0
    if pos_hits + neg_hits:
        sentiment_score = (pos_hits - neg_hits) / (pos_hits + neg_hits)

    currency_mentions = sorted({token.upper() for token in tokens if token in CURRENCY_CODES})
    mentions_sgd = any(term in tokens for term in SGD_KEY_TERMS)
    mas_mentions = tokens.count("mas")

    return {
        "story_id": news["story_id"],
        "headline": news["headline"],
        "published_at": news["published_at"],
        "source": news["source"],
        "word_count": token_count,
        "unique_word_count": unique_tokens,
        "sentiment_score": round(sentiment_score, 4),
        "positive_hits": pos_hits,
        "negative_hits": neg_hits,
        "mentions_sgd": int(mentions_sgd),
        "mas_mentions": mas_mentions,
        "currency_mentions": ",".join(currency_mentions),
        "path": news["path"],
    }


def process_once(args: argparse.Namespace, processed: Set[str]) -> bool:
    files = sorted(
        [
            path
            for path in args.input_dir.glob("**/*")
            if path.is_file() and not path.name.startswith(".")
        ]
    )
    new_rows: List[Dict[str, object]] = []
    updated = False

    for file_path in files:
        file_key = str(file_path.resolve())
        if file_key in processed:
            continue

        news = read_news(file_path)
        features = extract_features(news)
        new_rows.append(features)
        processed.add(file_key)
        updated = True

    if not new_rows:
        return updated

    df = pd.DataFrame(new_rows)
    args.silver_path.parent.mkdir(parents=True, exist_ok=True)
    header = not args.silver_path.exists()
    df.to_csv(args.silver_path, mode="a", header=header, index=False)

    if args.manifest:
        save_manifest(args.manifest, processed)

    return updated


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    args.input_dir.mkdir(parents=True, exist_ok=True)

    processed = load_manifest(args.manifest)

    updated = process_once(args, processed)
    if args.follow:
        try:
            while True:
                time.sleep(args.poll_interval)
                process_once(args, processed)
        except KeyboardInterrupt:
            pass
    elif not updated:
        sys.stderr.write("No new news files detected\n")


if __name__ == "__main__":
    main()
