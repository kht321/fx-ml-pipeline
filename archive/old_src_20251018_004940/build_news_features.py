"""Transform news Bronze data into Silver-level sentiment features.

This module is part of the News medallion pipeline. It processes raw news articles
from the Bronze layer using both lexicon-based and FinGPT-powered analysis to create
sophisticated sentiment and signal features for the Silver layer.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Optional

import pandas as pd

from fingpt_processor import create_processor, FinGPTAnalysis


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Define command-line interface for news feature engineering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/news/bronze/raw_articles"),
        help="Directory containing raw news files",
    )
    parser.add_argument(
        "--output-sentiment",
        type=Path,
        default=Path("data/news/silver/sentiment_scores/sentiment_features.csv"),
        help="CSV destination for sentiment analysis results",
    )
    parser.add_argument(
        "--output-entities",
        type=Path,
        default=Path("data/news/silver/entity_mentions/entity_features.csv"),
        help="CSV destination for named entity features",
    )
    parser.add_argument(
        "--output-topics",
        type=Path,
        default=Path("data/news/silver/topic_signals/topic_features.csv"),
        help="CSV destination for topic signal features",
    )
    parser.add_argument(
        "--use-fingpt",
        action="store_true",
        help="Use FinGPT for enhanced sentiment analysis (requires GPU)",
    )
    parser.add_argument(
        "--fingpt-model",
        default="FinGPT/fingpt-sentiment_llama2-7b_lora",
        help="FinGPT model to use for analysis",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of articles to process in each batch",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Watch input directory for new files",
    )
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=Path("data/news/bronze/.processed.json"),
        help="JSON file tracking processed articles",
    )
    parser.add_argument(
        "--market-features-path",
        type=Path,
        default=Path("data/market/silver/technical_features/sgd_vs_majors.csv"),
        help="Path to market technical features for context",
    )
    parser.add_argument(
        "--use-market-context",
        action="store_true",
        default=True,
        help="Include market context in FinGPT analysis",
    )
    return parser.parse_args(list(argv))


# Enhanced lexicons for financial analysis
FINANCIAL_POSITIVE = {
    'growth', 'gain', 'improve', 'strong', 'bullish', 'increase',
    'optimistic', 'upgrade', 'boost', 'surge', 'rally', 'outperform',
    'beat', 'exceed', 'record', 'robust', 'solid', 'resilient'
}

FINANCIAL_NEGATIVE = {
    'fall', 'risk', 'slowdown', 'bearish', 'decline', 'downgrade',
    'weak', 'loss', 'crash', 'plunge', 'drop', 'underperform',
    'miss', 'disappoint', 'concern', 'volatile', 'uncertainty'
}

VOLATILITY_KEYWORDS = {
    'volatile', 'volatility', 'turbulent', 'unstable', 'erratic',
    'swing', 'whipsaw', 'choppy', 'wild', 'dramatic'
}

SGD_KEYWORDS = {
    'sgd', 'singapore dollar', 'monetary authority', 'mas', 'singapore',
    'sing dollar', 's$', 'singapore monetary authority'
}

CURRENCY_CODES = {
    'usd', 'sgd', 'eur', 'gbp', 'jpy', 'aud', 'chf', 'cny',
    'cad', 'nzd', 'hkd', 'krw', 'inr', 'thb', 'myr'
}

POLICY_HAWKISH = {
    'tighten', 'tightening', 'hawkish', 'aggressive', 'restrictive',
    'raise rates', 'rate hike', 'combat inflation', 'cool economy'
}

POLICY_DOVISH = {
    'ease', 'easing', 'dovish', 'accommodative', 'stimulus',
    'cut rates', 'rate cut', 'support growth', 'boost economy'
}


def get_latest_market_context(market_features_path: Path, target_instrument: str = "USD_SGD") -> Dict:
    """Get the most recent market features for news analysis context.

    Parameters
    ----------
    market_features_path : Path
        Path to the market technical features CSV
    target_instrument : str
        Instrument to get context for (default: USD_SGD)

    Returns
    -------
    dict
        Market context with key technical indicators
    """
    try:
        if not market_features_path.exists():
            log(f"Market features file not found: {market_features_path}")
            return {}

        # Read the market features file
        market_df = pd.read_csv(market_features_path)

        if market_df.empty:
            log("Market features file is empty")
            return {}

        # Filter for target instrument and get latest observation
        instrument_data = market_df[market_df['instrument'] == target_instrument]

        if instrument_data.empty:
            log(f"No data found for instrument: {target_instrument}")
            return {}

        # Get the most recent row
        latest = instrument_data.iloc[-1]

        # Extract key market context features
        market_context = {
            'mid': latest.get('mid', 0.0),
            'ret_5': latest.get('ret_5', 0.0),
            'vol_20': latest.get('roll_vol_20', latest.get('vol_20', 0.0)),
            'high_vol_regime': latest.get('high_vol_regime', False),
            'spread_pct': latest.get('spread_pct', latest.get('spread', 0.0) / latest.get('mid', 1.0)),
            'zscore_20': latest.get('zscore_20', 0.0),
            'session': determine_trading_session(latest.get('time', pd.Timestamp.now())),
            'timestamp': latest.get('time', pd.Timestamp.now())
        }

        log(f"Retrieved market context for {target_instrument}: mid={market_context['mid']:.4f}, "
            f"vol={market_context['vol_20']:.2%}, regime={'High' if market_context['high_vol_regime'] else 'Normal'}")

        return market_context

    except Exception as e:
        log(f"Failed to get market context: {e}")
        return {}


def determine_trading_session(timestamp) -> str:
    """Determine trading session based on timestamp."""
    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        hour = timestamp.hour

        if 0 <= hour < 8:
            return "Asian"
        elif 8 <= hour < 16:
            return "London"
        elif 16 <= hour < 24:
            return "New York"
        else:
            return "Unknown"
    except:
        return "Unknown"


class NewsProcessor:
    """Enhanced news processing with FinGPT integration."""

    def __init__(self, use_fingpt: bool = False, fingpt_model: str = None):
        """Initialize the news processor.

        Parameters
        ----------
        use_fingpt : bool
            Whether to use FinGPT for enhanced analysis
        fingpt_model : str
            FinGPT model name to use
        """
        self.use_fingpt = use_fingpt

        # Initialize the appropriate processor
        if use_fingpt:
            self.processor = create_processor(
                use_fingpt=True,
                model_name=fingpt_model or "FinGPT/fingpt-sentiment_llama2-7b_lora"
            )
        else:
            self.processor = create_processor(use_fingpt=False)

    def load_processed_manifest(self, manifest_path: Path) -> Set[str]:
        """Load the set of already-processed file paths."""
        if not manifest_path.exists():
            return set()

        try:
            with manifest_path.open('r') as f:
                data = json.load(f)
                return set(data.get('processed_files', []))
        except (json.JSONDecodeError, KeyError):
            return set()

    def save_processed_manifest(self, manifest_path: Path, processed_files: Set[str]):
        """Save the updated set of processed files."""
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'processed_files': list(processed_files),
            'last_updated': pd.Timestamp.now().isoformat()
        }

        with manifest_path.open('w') as f:
            json.dump(data, f, indent=2)

    def parse_article_file(self, file_path: Path) -> Dict:
        """Parse a news article file (JSON or plain text)."""
        try:
            if file_path.suffix.lower() == '.json':
                with file_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)

                return {
                    'story_id': data.get('story_id', file_path.stem),
                    'headline': data.get('headline', ''),
                    'body': data.get('body', ''),
                    'published_at': data.get('published_at', file_path.stat().st_mtime),
                    'source': data.get('source', 'unknown'),
                    'file_path': str(file_path)
                }
            else:
                # Plain text file
                with file_path.open('r', encoding='utf-8') as f:
                    content = f.read().strip()

                return {
                    'story_id': file_path.stem,
                    'headline': content[:100] + '...' if len(content) > 100 else content,
                    'body': content,
                    'published_at': file_path.stat().st_mtime,
                    'source': 'text_file',
                    'file_path': str(file_path)
                }

        except Exception as e:
            print(f"Error parsing {file_path}: {e}", file=sys.stderr)
            return None

    def extract_lexicon_features(self, text: str) -> Dict:
        """Extract basic lexicon-based features."""
        text_lower = text.lower()
        words = set(text_lower.split())

        # Basic sentiment counts
        positive_hits = len(words & FINANCIAL_POSITIVE)
        negative_hits = len(words & FINANCIAL_NEGATIVE)
        volatility_hits = len(words & VOLATILITY_KEYWORDS)

        # Policy sentiment
        hawkish_hits = len(words & POLICY_HAWKISH)
        dovish_hits = len(words & POLICY_DOVISH)

        # SGD relevance
        sgd_mentions = any(keyword in text_lower for keyword in SGD_KEYWORDS)
        currency_mentions = list(words & CURRENCY_CODES)

        # Simple sentiment score
        total_words = len(text.split())
        if positive_hits > negative_hits:
            sentiment_score = min(1.0, (positive_hits - negative_hits) / max(total_words, 1))
        elif negative_hits > positive_hits:
            sentiment_score = max(-1.0, (positive_hits - negative_hits) / max(total_words, 1))
        else:
            sentiment_score = 0.0

        return {
            'word_count': len(text.split()),
            'unique_word_count': len(words),
            'positive_hits': positive_hits,
            'negative_hits': negative_hits,
            'volatility_hits': volatility_hits,
            'hawkish_hits': hawkish_hits,
            'dovish_hits': dovish_hits,
            'sentiment_score_lexicon': sentiment_score,
            'mentions_sgd': sgd_mentions,
            'currency_mentions': ','.join(currency_mentions) if currency_mentions else '',
            'policy_tone': 'hawkish' if hawkish_hits > dovish_hits else 'dovish' if dovish_hits > hawkish_hits else 'neutral'
        }

    def process_article(self, article_data: Dict, market_context: Dict = None) -> Dict:
        """Process a single article using both lexicon and FinGPT analysis."""
        if not article_data:
            return None

        # Basic article metadata
        result = {
            'story_id': article_data['story_id'],
            'headline': article_data['headline'],
            'body': article_data['body'],
            'published_at': pd.to_datetime(article_data['published_at']),
            'source': article_data['source'],
            'file_path': article_data['file_path']
        }

        # Combine headline and body for analysis
        full_text = f"{article_data['headline']} {article_data['body']}"

        # Extract lexicon-based features
        lexicon_features = self.extract_lexicon_features(full_text)
        result.update(lexicon_features)

        # FinGPT analysis if enabled
        if self.use_fingpt and hasattr(self.processor, 'analyze_sgd_news'):
            try:
                fingpt_analysis = self.processor.analyze_sgd_news(
                    article_data['body'],
                    article_data['headline'],
                    market_context=market_context  # Pass market context to FinGPT
                )

                # Add FinGPT features including new market-aware fields
                result.update({
                    'sentiment_score_fingpt': fingpt_analysis.sentiment_score,
                    'confidence_fingpt': fingpt_analysis.confidence,
                    'sgd_directional_signal': fingpt_analysis.sgd_directional_signal,
                    'policy_implications': fingpt_analysis.policy_implications,
                    'time_horizon': fingpt_analysis.time_horizon,
                    'key_factors': ';'.join(fingpt_analysis.key_factors),
                    'market_coherence': fingpt_analysis.market_coherence,
                    'signal_strength_adjusted': fingpt_analysis.signal_strength_adjusted,
                    'fingpt_raw_response': fingpt_analysis.raw_response[:500]  # Truncate for storage
                })

                # Add market context metadata if available
                if market_context:
                    result.update({
                        'market_mid_price': market_context.get('mid', 0.0),
                        'market_volatility': market_context.get('vol_20', 0.0),
                        'market_session': market_context.get('session', 'Unknown'),
                        'market_vol_regime': market_context.get('high_vol_regime', False)
                    })

                # Use FinGPT sentiment as primary if available
                result['sentiment_score'] = fingpt_analysis.sentiment_score
                result['confidence'] = fingpt_analysis.confidence

            except Exception as e:
                print(f"FinGPT analysis failed for {result['story_id']}: {e}", file=sys.stderr)
                # Fallback to lexicon features
                result['sentiment_score'] = lexicon_features['sentiment_score_lexicon']
                result['confidence'] = 0.3  # Low confidence for lexicon fallback
        else:
            # Use lexicon sentiment as primary
            result['sentiment_score'] = lexicon_features['sentiment_score_lexicon']
            result['confidence'] = 0.3

        return result

    def create_feature_splits(self, processed_articles: List[Dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split processed articles into different feature categories."""
        if not processed_articles:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(processed_articles)

        # Sentiment features
        sentiment_cols = [
            'story_id', 'headline', 'published_at', 'source',
            'sentiment_score', 'confidence', 'sentiment_score_lexicon',
            'positive_hits', 'negative_hits', 'volatility_hits',
            'policy_tone', 'hawkish_hits', 'dovish_hits'
        ]

        # Add FinGPT columns if they exist
        fingpt_cols = ['sentiment_score_fingpt', 'confidence_fingpt', 'sgd_directional_signal',
                      'policy_implications', 'time_horizon']
        sentiment_cols.extend([col for col in fingpt_cols if col in df.columns])

        sentiment_df = df[[col for col in sentiment_cols if col in df.columns]].copy()

        # Entity/mention features
        entity_cols = [
            'story_id', 'published_at', 'mentions_sgd', 'currency_mentions',
            'word_count', 'unique_word_count'
        ]
        if 'key_factors' in df.columns:
            entity_cols.append('key_factors')

        entity_df = df[[col for col in entity_cols if col in df.columns]].copy()

        # Topic signal features (aggregate sentiment by topic)
        topic_df = df[['story_id', 'published_at', 'source', 'sentiment_score', 'mentions_sgd']].copy()
        topic_df['topic_category'] = 'financial'  # Could be enhanced with topic modeling
        topic_df['relevance_score'] = topic_df['mentions_sgd'].astype(float)

        return sentiment_df, entity_df, topic_df


def append_to_csv(df: pd.DataFrame, output_path: Path) -> int:
    """Append new rows to CSV file."""
    if df.empty:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with header if file doesn't exist
    write_header = not output_path.exists()
    df.to_csv(output_path, mode='a', header=write_header, index=False)

    return len(df)


def log(message: str) -> None:
    """Emit structured progress messages to stderr."""
    sys.stderr.write(f"[build_news_features] {message}\n")
    sys.stderr.flush()


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Main processing loop for news feature engineering."""
    args = parse_args(argv or sys.argv[1:])

    # Initialize processor
    processor = NewsProcessor(
        use_fingpt=args.use_fingpt,
        fingpt_model=args.fingpt_model
    )

    log(f"Starting news feature engineering ({'FinGPT' if args.use_fingpt else 'lexicon'} mode)")

    # Load processed manifest
    processed_files = processor.load_processed_manifest(args.processed_manifest)

    def process_batch():
        """Process a batch of new files."""
        # Find new files
        input_files = []
        for pattern in ['*.txt', '*.json']:
            input_files.extend(args.input_dir.glob(pattern))

        new_files = [f for f in input_files if str(f) not in processed_files]

        if not new_files:
            return 0

        log(f"Processing {len(new_files)} new files")

        # Get market context if enabled
        market_context = None
        if args.use_market_context:
            market_context = get_latest_market_context(args.market_features_path)
            if market_context:
                log(f"Using market context: {market_context['session']} session, "
                    f"vol regime: {'High' if market_context['high_vol_regime'] else 'Normal'}")
            else:
                log("No market context available - proceeding without")

        # Process files in batches
        total_processed = 0

        for i in range(0, len(new_files), args.batch_size):
            batch_files = new_files[i:i + args.batch_size]
            batch_articles = []

            for file_path in batch_files:
                article_data = processor.parse_article_file(file_path)
                if article_data:
                    processed_article = processor.process_article(article_data, market_context)
                    if processed_article:
                        batch_articles.append(processed_article)

                # Mark as processed
                processed_files.add(str(file_path))

            if batch_articles:
                # Split into feature categories
                sentiment_df, entity_df, topic_df = processor.create_feature_splits(batch_articles)

                # Write to separate files
                sentiment_written = append_to_csv(sentiment_df, args.output_sentiment)
                entity_written = append_to_csv(entity_df, args.output_entities)
                topic_written = append_to_csv(topic_df, args.output_topics)

                total_processed += len(batch_articles)
                log(f"Batch complete: {len(batch_articles)} articles, "
                    f"{sentiment_written} sentiment, {entity_written} entity, {topic_written} topic rows")

        # Save updated manifest
        processor.save_processed_manifest(args.processed_manifest, processed_files)

        return total_processed

    try:
        if args.follow:
            log("Watching for new files (Ctrl+C to stop)")
            while True:
                processed_count = process_batch()
                if processed_count > 0:
                    log(f"Processed {processed_count} articles")
                time.sleep(5)  # Check every 5 seconds
        else:
            processed_count = process_batch()
            log(f"Finished: {processed_count} articles processed")

    except KeyboardInterrupt:
        log("Interrupted by user")
    except Exception as e:
        log(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()