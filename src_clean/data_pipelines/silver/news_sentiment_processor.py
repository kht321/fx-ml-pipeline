"""
News Sentiment Features Processor - Silver Layer

Repository Location: fx-ml-pipeline/src_clean/data_pipelines/silver/news_sentiment_processor.py

Purpose:
    Processes bronze layer news articles into sentiment features using lexicon-based analysis.
    Computes sentiment scores, polarity, subjectivity, and financial tone indicators.

Input:
    - Bronze news data: data_clean/bronze/news/*.json

Output:
    - Silver sentiment features: data_clean/silver/news/sentiment/*.csv
    - Features: Sentiment score, polarity, confidence, financial tone

Usage:
    python src_clean/data_pipelines/silver/news_sentiment_processor.py \
        --input-dir data_clean/bronze/news \
        --output data_clean/silver/news/sentiment/spx500_sentiment.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import re

import pandas as pd
from textblob import TextBlob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentProcessor:
    """Processes news articles into sentiment features."""

    def __init__(self, input_dir: Path, output_path: Path):
        self.input_dir = input_dir
        self.output_path = output_path

        # Financial sentiment lexicon (simplified)
        self.positive_words = {
            'rally', 'surge', 'gain', 'jump', 'rise', 'soar', 'climb', 'advance',
            'bullish', 'optimistic', 'growth', 'profit', 'beat', 'strong', 'robust'
        }

        self.negative_words = {
            'plunge', 'crash', 'drop', 'fall', 'decline', 'sink', 'slide', 'tumble',
            'bearish', 'pessimistic', 'loss', 'miss', 'weak', 'concern', 'worry'
        }

        self.hawkish_words = {
            'hawkish', 'tighten', 'raise rates', 'inflation', 'restrictive', 'hike'
        }

        self.dovish_words = {
            'dovish', 'ease', 'cut rates', 'accommodative', 'stimulus', 'support'
        }

    def load_articles(self) -> List[Dict]:
        """Load all articles from bronze layer."""
        logger.info(f"Loading articles from {self.input_dir}")

        articles = []
        for json_file in self.input_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    article = json.load(f)
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Could not load {json_file}: {e}")

        logger.info(f"Loaded {len(articles)} articles")
        return articles

    def compute_textblob_sentiment(self, text: str) -> tuple:
        """Compute TextBlob sentiment (polarity and subjectivity)."""
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def compute_financial_sentiment(self, text: str) -> float:
        """Compute financial-specific sentiment score."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def compute_policy_tone(self, text: str) -> str:
        """Determine monetary policy tone (hawkish/dovish/neutral)."""
        text_lower = text.lower()

        hawkish_score = sum(1 for word in self.hawkish_words if word in text_lower)
        dovish_score = sum(1 for word in self.dovish_words if word in text_lower)

        if hawkish_score > dovish_score:
            return 'hawkish'
        elif dovish_score > hawkish_score:
            return 'dovish'
        else:
            return 'neutral'

    def process_article(self, article: Dict) -> Dict:
        """Process a single article into sentiment features."""
        # Combine headline and body
        text = f"{article.get('headline', '')} {article.get('body', '')}"

        # TextBlob sentiment
        polarity, subjectivity = self.compute_textblob_sentiment(text)

        # Financial sentiment
        financial_sentiment = self.compute_financial_sentiment(text)

        # Policy tone
        policy_tone = self.compute_policy_tone(text)

        # Confidence (inverse of subjectivity, high confidence = objective reporting)
        confidence = 1.0 - subjectivity

        return {
            'article_id': article.get('article_id', article.get('story_id', 'unknown')),
            'published_at': article.get('published_at'),
            'source': article.get('source'),
            'headline': article.get('headline', '')[:100],  # Truncate for storage
            'polarity': float(polarity),
            'subjectivity': float(subjectivity),
            'financial_sentiment': float(financial_sentiment),
            'confidence': float(confidence),
            'policy_tone': policy_tone,
            'headline_length': len(article.get('headline', '')),
            'body_length': len(article.get('body', ''))
        }

    def process_all_articles(self, articles: List[Dict]) -> pd.DataFrame:
        """Process all articles into sentiment features."""
        logger.info("Computing sentiment features...")

        processed = []
        for article in articles:
            try:
                features = self.process_article(article)
                processed.append(features)
            except Exception as e:
                logger.warning(f"Error processing article: {e}")

        df = pd.DataFrame(processed)

        # Convert published_at to datetime
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce')

        # Sort by time
        df = df.sort_values('published_at').reset_index(drop=True)

        logger.info(f"Processed {len(df)} articles with sentiment features")
        return df

    def save_features(self, df: pd.DataFrame):
        """Save sentiment features to silver layer."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved {len(df)} sentiment features to {self.output_path}")

    def run(self):
        """Execute processing pipeline."""
        logger.info("="*80)
        logger.info("Sentiment Features Processor - Silver Layer")
        logger.info("="*80)

        articles = self.load_articles()
        if not articles:
            logger.error("No articles to process")
            return

        features_df = self.process_all_articles(articles)
        if features_df.empty:
            logger.error("No features computed")
            return

        self.save_features(features_df)

        logger.info("="*80)
        logger.info("Sentiment features processing complete")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Bronze news directory")
    parser.add_argument("--output", type=Path, required=True, help="Silver CSV file")
    args = parser.parse_args()

    processor = SentimentProcessor(args.input_dir, args.output)
    processor.run()


if __name__ == "__main__":
    main()
