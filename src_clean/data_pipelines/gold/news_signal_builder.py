"""
News Signal Builder - Gold Layer

Repository Location: fx-ml-pipeline/src_clean/data_pipelines/gold/news_signal_builder.py

Purpose:
    Transforms silver layer sentiment into trading signals using FinBERT.
    Aggregates news over time windows and generates signals for training pipeline.

Input:
    - Silver sentiment: data_clean/silver/news/sentiment/*.csv (TextBlob output)
    - Bronze news: data_clean/bronze/news/**/*.json (for article bodies)

Output:
    - Gold signals: data_clean/gold/news/signals/*.csv
    - Features: signal_time, avg_sentiment, signal_strength, trading_signal, etc.

Usage:
    python src_clean/data_pipelines/gold/news_signal_builder.py \
        --silver-sentiment data_clean/silver/news/sentiment/spx500_sentiment.csv \
        --bronze-news data_clean/bronze/news \
        --output data_clean/gold/news/signals/spx500_news_signals.csv \
        --window 60  # 60 minute aggregation window
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import timedelta

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinBERTSignalBuilder:
    """Builds trading signals using FinBERT financial sentiment analysis."""

    def __init__(self, aggregation_window_minutes: int = 60):
        self.window = aggregation_window_minutes
        self.tokenizer = None
        self.model = None
        self.device = None

    def load_finbert(self):
        """Load FinBERT model for financial sentiment."""
        logger.info("Loading FinBERT model (ProsusAI/finbert)...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

            # Use GPU if available, otherwise CPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"FinBERT loaded successfully on {self.device}")
            logger.info(f"Model size: ~209 MB")

        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            raise

    def analyze_with_finbert(self, text: str) -> Dict:
        """
        Run FinBERT sentiment analysis on text.

        Args:
            text: Article headline + body

        Returns:
            dict with sentiment, confidence, and class scores
        """
        # Truncate to FinBERT's max length (512 tokens)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # FinBERT outputs: [positive, negative, neutral]
        scores = probs[0].cpu().numpy()
        labels = ['positive', 'negative', 'neutral']

        sentiment_dict = dict(zip(labels, scores))
        predicted_label = max(sentiment_dict, key=sentiment_dict.get)
        confidence = sentiment_dict[predicted_label]

        # Convert to numeric sentiment score (-1 to 1)
        # positive - negative, weighted by confidence
        sentiment_score = float(scores[0] - scores[1])

        return {
            'sentiment': predicted_label,
            'sentiment_score': sentiment_score,
            'confidence': float(confidence),
            'positive_prob': float(scores[0]),
            'negative_prob': float(scores[1]),
            'neutral_prob': float(scores[2])
        }

    def load_article_bodies(self, bronze_news_dir: Path) -> Dict[str, str]:
        """
        Load full article bodies from bronze layer.

        Args:
            bronze_news_dir: Bronze news directory

        Returns:
            Dict mapping article_id to full article text
        """
        logger.info(f"Loading article bodies from bronze: {bronze_news_dir}")

        article_bodies = {}

        # Check both root and subdirectories (e.g., hybrid/)
        json_files = list(bronze_news_dir.glob("*.json"))
        json_files.extend(bronze_news_dir.glob("*/*.json"))

        for json_file in json_files:
            # Skip tracking files
            if json_file.name == "seen_articles.json":
                continue

            try:
                with open(json_file, 'r') as f:
                    article = json.load(f)

                article_id = article.get('article_id') or article.get('story_id')
                if not article_id:
                    continue

                # Combine headline and body
                headline = article.get('headline', '')
                body = article.get('body', '') or article.get('summary', '')
                full_text = f"{headline} {body}"

                article_bodies[article_id] = full_text

            except Exception as e:
                logger.warning(f"Error loading {json_file.name}: {e}")

        logger.info(f"Loaded {len(article_bodies)} article bodies")
        return article_bodies

    def process_articles(self, silver_df: pd.DataFrame, article_bodies: Dict[str, str]) -> pd.DataFrame:
        """Process all articles with FinBERT."""
        logger.info(f"Processing {len(silver_df)} articles with FinBERT...")

        results = []
        skipped = 0

        for idx, row in tqdm(silver_df.iterrows(), total=len(silver_df), desc="FinBERT Analysis"):
            article_id = row.get('article_id')

            # Get full article text from bronze
            full_text = article_bodies.get(article_id)

            if not full_text or len(full_text.strip()) < 10:
                # Fallback to silver layer text if bronze not available
                full_text = f"{row.get('headline', '')} {row.get('body', '')}"

            if len(full_text.strip()) < 10:
                skipped += 1
                continue

            # Run FinBERT
            try:
                finbert_result = self.analyze_with_finbert(full_text)
            except Exception as e:
                logger.warning(f"FinBERT analysis failed for article {article_id}: {e}")
                skipped += 1
                continue

            # Combine with original data
            result = {
                'article_id': article_id,
                'published_at': row.get('published_at'),
                'source': row.get('source'),
                'headline': row.get('headline', '')[:100],
                **finbert_result
            }

            results.append(result)

        if skipped > 0:
            logger.warning(f"Skipped {skipped} articles due to errors or missing text")

        results_df = pd.DataFrame(results)
        results_df['published_at'] = pd.to_datetime(results_df['published_at'], utc=True)

        logger.info(f"FinBERT analysis complete: {len(results_df)} articles processed")

        # Distribution stats
        sentiment_counts = results_df['sentiment'].value_counts()
        logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
        logger.info(f"Avg sentiment score: {results_df['sentiment_score'].mean():.3f}")
        logger.info(f"Avg confidence: {results_df['confidence'].mean():.3f}")

        return results_df

    def aggregate_signals(self, finbert_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate FinBERT results into time-windowed trading signals.

        Aggregates articles over rolling windows (e.g., 60 minutes) to create
        stable trading signals expected by the training pipeline.
        """
        logger.info(f"Aggregating signals over {self.window}-minute windows...")

        # Sort by time
        finbert_df = finbert_df.sort_values('published_at').reset_index(drop=True)

        # Create time windows
        finbert_df['window_start'] = finbert_df['published_at'].dt.floor(f'{self.window}min')

        # Aggregate by window
        aggregated = finbert_df.groupby('window_start').agg({
            'sentiment_score': 'mean',      # avg_sentiment
            'confidence': 'mean',           # quality_score
            'article_id': 'count',          # article_count
            'positive_prob': 'mean',
            'negative_prob': 'mean',
            'neutral_prob': 'mean',
            'headline': 'last',             # Latest headline in window
            'source': 'last'                # Latest source in window
        }).reset_index()

        # Rename columns for training pipeline
        aggregated.rename(columns={
            'window_start': 'signal_time',
            'sentiment_score': 'avg_sentiment',
            'confidence': 'quality_score',
            'article_id': 'article_count',
            'headline': 'latest_headline',
            'source': 'latest_source'
        }, inplace=True)

        # Compute signal strength (absolute sentiment * confidence)
        aggregated['signal_strength'] = (
            aggregated['avg_sentiment'].abs() * aggregated['quality_score']
        )

        # Generate trading signal based on sentiment
        def compute_trading_signal(row):
            """
            Convert sentiment to trading signal.

            Returns:
                1: Buy/bullish (positive sentiment, strong signal)
                -1: Sell/bearish (negative sentiment, strong signal)
                0: Hold/neutral (weak or neutral sentiment)
            """
            sentiment = row['avg_sentiment']
            strength = row['signal_strength']

            if sentiment > 0.2 and strength > 0.3:
                return 1  # Buy
            elif sentiment < -0.2 and strength > 0.3:
                return -1  # Sell
            else:
                return 0  # Hold

        aggregated['trading_signal'] = aggregated.apply(compute_trading_signal, axis=1)

        logger.info(f"Created {len(aggregated)} aggregated signals")

        # Signal distribution
        buy_count = sum(aggregated['trading_signal'] == 1)
        sell_count = sum(aggregated['trading_signal'] == -1)
        hold_count = sum(aggregated['trading_signal'] == 0)

        logger.info(f"Trading signals: Buy={buy_count}, Sell={sell_count}, Hold={hold_count}")
        logger.info(f"Avg sentiment: {aggregated['avg_sentiment'].mean():.3f}")
        logger.info(f"Avg signal strength: {aggregated['signal_strength'].mean():.3f}")

        return aggregated

    def run(self, silver_path: Path, bronze_dir: Path, output_path: Path):
        """Execute the full pipeline."""
        logger.info("="*80)
        logger.info("News Signal Builder - Gold Layer (FinBERT)")
        logger.info("="*80)

        # Load silver sentiment data
        logger.info(f"Loading silver sentiment from: {silver_path}")
        silver_df = pd.read_csv(silver_path)
        silver_df['published_at'] = pd.to_datetime(silver_df['published_at'], utc=True)
        logger.info(f"Loaded {len(silver_df)} articles from silver")

        # Load article bodies from bronze
        article_bodies = self.load_article_bodies(bronze_dir)

        # Load FinBERT
        self.load_finbert()

        # Process with FinBERT
        finbert_df = self.process_articles(silver_df, article_bodies)

        if finbert_df.empty:
            logger.error("No articles processed successfully. Exiting.")
            return

        # Aggregate into signals
        signals_df = self.aggregate_signals(finbert_df)

        # Save gold signals
        output_path.parent.mkdir(parents=True, exist_ok=True)
        signals_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(signals_df)} signals to: {output_path}")

        # Summary
        logger.info("\n" + "="*80)
        logger.info("GOLD LAYER BUILD COMPLETE")
        logger.info("="*80)
        logger.info(f"Input articles: {len(silver_df)}")
        logger.info(f"Processed articles: {len(finbert_df)}")
        logger.info(f"Output signals: {len(signals_df)}")
        logger.info(f"Time range: {signals_df['signal_time'].min()} to {signals_df['signal_time'].max()}")
        logger.info(f"Features generated:")
        logger.info(f"  - signal_time")
        logger.info(f"  - avg_sentiment ({signals_df['avg_sentiment'].mean():.3f} ± {signals_df['avg_sentiment'].std():.3f})")
        logger.info(f"  - signal_strength ({signals_df['signal_strength'].mean():.3f} ± {signals_df['signal_strength'].std():.3f})")
        logger.info(f"  - trading_signal (Buy/Sell/Hold)")
        logger.info(f"  - article_count ({signals_df['article_count'].sum()} total)")
        logger.info(f"  - quality_score ({signals_df['quality_score'].mean():.3f})")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silver-sentiment",
        type=Path,
        required=True,
        help="Silver sentiment CSV (from TextBlob processor)"
    )
    parser.add_argument(
        "--bronze-news",
        type=Path,
        required=True,
        help="Bronze news directory (for full article bodies)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output gold signals CSV"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Aggregation window in minutes (default: 60)"
    )
    args = parser.parse_args()

    builder = FinBERTSignalBuilder(aggregation_window_minutes=args.window)
    builder.run(args.silver_sentiment, args.bronze_news, args.output)


if __name__ == "__main__":
    main()
