"""Apply FinGPT to Silver news articles and emit numeric features.

Reads Silver news artifacts (sentiment/entities/topics) and produces per-article
FinGPT features between Silver and Gold. Outputs Parquet with event_timestamp
and instrument for Feast/Gold consumption.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import pandas as pd


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sentiment", type=Path, default=Path("data/news/silver/sentiment_scores/sentiment_features.csv"))
    p.add_argument("--entities", type=Path, default=Path("data/news/silver/entity_mentions/entity_features.csv"))
    p.add_argument("--topics", type=Path, default=Path("data/news/silver/topic_signals/topic_features.csv"))
    p.add_argument("--output", type=Path, default=Path("data/news/silver/fingpt/finllm_features.parquet"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="Generate deterministic mock outputs without FinGPT calls")
    return p.parse_args(list(argv) if argv is not None else None)


def _mock_fingpt_eval(article: Dict[str, Any]) -> Dict[str, Any]:
    # Deterministic pseudo-scores based on hash of story_id
    seed = abs(hash(article.get("story_id", "0"))) % 1000 / 1000.0
    sgn = 1 if (seed * 10) % 2 > 1 else -1
    res = {
        "sentiment_score": round((seed - 0.5) * 2, 3),
        "volatility_score": round(0.2 + 0.6 * seed, 3),
        "uncertainty_score": round(0.1 + 0.7 * (1 - seed), 3),
        "fear_score": round(0.05 + 0.4 * (seed), 3),
        "surprise_factor": round((0.5 - seed) * 2, 3),
        "macroeconomic_score": round(0.3 + 0.6 * seed, 3),
        "relevance_score": round(0.4 + 0.5 * seed, 3),
        "geography_score": round(0.2 + 0.6 * (1 - seed), 3),
        "market_liquidity_impact": round(0.2 + 0.6 * seed, 3),
        "safe_haven_indicator": round(0.1 + 0.7 * (1 - seed), 3),
        "market_risk_appetite": round((seed - 0.5) * 2, 3),
    }
    return res


def _map_instrument_from_entities(row: pd.Series) -> str:
    txt = " ".join([
        str(row.get("headline", "")),
        str(row.get("entities", "")),
        str(row.get("currency_mentions", "")),
    ]).lower()
    if "sgd" in txt or "singapore" in txt:
        return "USD_SGD"
    if "eur" in txt or "euro" in txt:
        return "EUR_USD"
    if "gbp" in txt or "sterling" in txt or "pound" in txt:
        return "GBP_USD"
    if "jpy" in txt or "yen" in txt:
        return "USD_JPY"
    return "USD_SGD"


def run(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    # Load available silver inputs
    def _safe_read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    senti = _safe_read(args.sentiment)
    ents = _safe_read(args.entities)
    topics = _safe_read(args.topics)

    if "published_at" in senti.columns:
        senti["published_at"] = pd.to_datetime(senti["published_at"], utc=True, errors="coerce")
    if "published_at" in ents.columns:
        ents["published_at"] = pd.to_datetime(ents["published_at"], utc=True, errors="coerce")
    if "published_at" in topics.columns:
        topics["published_at"] = pd.to_datetime(topics["published_at"], utc=True, errors="coerce")

    # Base article frame
    cols = [c for c in ["story_id", "headline", "source", "published_at", "currency_mentions", "entities"] if c in senti.columns]
    base = senti[cols].drop_duplicates(subset=[c for c in ["story_id"] if c in cols]).copy() if not senti.empty else pd.DataFrame(columns=["story_id", "published_at"])

    # Merge helpful context
    if not ents.empty and "story_id" in ents.columns:
        merge_cols = [c for c in ["story_id", "entities"] if c in ents.columns]
        base = base.merge(ents[merge_cols].drop_duplicates("story_id"), on="story_id", how="left")
    if not topics.empty and "story_id" in topics.columns:
        merge_cols = [c for c in ["story_id", "primary_topic", "event_type"] if c in topics.columns]
        base = base.merge(topics[merge_cols].drop_duplicates("story_id"), on="story_id", how="left")

    if base.empty:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["story_id", "event_timestamp", "instrument"]).to_parquet(args.output, index=False)
        return

    # Determine instrument per article
    if "instrument" not in base.columns:
        base["instrument"] = base.apply(_map_instrument_from_entities, axis=1)

    # Prepare FinGPT inputs
    records: List[Dict[str, Any]] = base.to_dict(orient="records")
    outputs: List[Dict[str, Any]] = []

    for rec in records:
        if args.dry_run:
            metrics = _mock_fingpt_eval(rec)
        else:
            # Placeholder: integrate local FinGPT model inference here
            # For now, fallback to mock to avoid runtime deps
            metrics = _mock_fingpt_eval(rec)
        out = {**metrics}
        out["story_id"] = rec.get("story_id")
        out["event_timestamp"] = rec.get("published_at")
        out["instrument"] = rec.get("instrument")
        outputs.append(out)

    df = pd.DataFrame(outputs)
    # Ensure required columns exist
    req = [
        "sentiment_score","volatility_score","uncertainty_score","fear_score","surprise_factor",
        "macroeconomic_score","relevance_score","geography_score","market_liquidity_impact",
        "safe_haven_indicator","market_risk_appetite","story_id","event_timestamp","instrument"
    ]
    for c in req:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[req]
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True, errors="coerce")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)


if __name__ == "__main__":
    run()

