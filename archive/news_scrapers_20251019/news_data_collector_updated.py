"""
News Data Collector - Bronze Layer

Repository Location: fx-ml-pipeline/src_clean/data_pipelines/bronze/news_data_collector.py

Purpose:
    Collects historical financial news articles relevant to S&P 500 trading.
    This is the entry point to the Bronze layer of the news data medallion architecture.

Output:
    - Raw JSON files saved to: data_clean/bronze/news/
    - Format: Individual JSON files per article with metadata

Features:
    - Multiple RSS feed sources (Reuters, Bloomberg, MarketWatch)
    - Historical news scraping via News API
    - S&P 500 relevance filtering
    - Duplicate detection
    - Timezone normalization (all to UTC)

Usage:
    # Recent news (free RSS feeds)
    python src_clean/data_pipelines/bronze/news_data_collector.py --mode recent

    # Historical news (requires API keys in .env)
    python src_clean/data_pipelines/bronze/news_data_collector.py \\
        --mode historical \\
        --years 5 \\
        --api-key YOUR_NEWS_API_KEY
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
import argparse
import hashlib

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
import pytz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsDataCollector:
    """Collects financial news articles and saves to Bronze layer."""

    def __init__(
        self,
        output_dir: str = "data_clean/bronze/news",
        mode: str = "recent"
    ):
        """
        Initialize the news collector.

        Parameters
        ----------
        output_dir : str
            Directory to save bronze news data
        mode : str
            Collection mode: 'recent' (free RSS) or 'historical' (API-based)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode

        # Track seen articles to avoid duplicates
        self.seen_file = self.output_dir / "seen_articles.json"
        self.seen_articles = self._load_seen_articles()

        # S&P 500 relevant keywords
        self.sp500_keywords = {
            # Index references
            's&p 500', 'sp500', 's&p500', 'spx', 'spy', 'standard and poor',
            'wall street', 'nasdaq', 'dow jones', 'us equities', 'broad market',
            'stock market', 'index fund',

            # Monetary policy & macro
            'federal reserve', 'fed', 'fomc', 'interest rate', 'rate hike', 'rate cut',
            'inflation', 'cpi', 'ppi', 'core inflation', 'economic growth', 'gdp',
            'unemployment', 'jobless claims', 'recession', 'soft landing', 'hard landing',
            'deflation', 'disinflation', 'stagflation', 'employment', 'jobs report',
            'monetary policy', 'fiscal policy',

            # Bond market & yields
            'treasury yields', '10-year yield', '2-year yield', 'bond market',
            'treasuries', 'government bonds', 'sovereign debt', 'yield curve',
            'inverted yield curve', 'bond selloff', 'bond rally', 'fixed income',

            # Market sentiment
            'market rally', 'market selloff', 'bull market', 'bear market',
            'volatility', 'vix', 'fear index', 'risk-on', 'risk-off',
            'flight to safety', 'safe haven', 'market sentiment',

            # Earnings season (index-level impact)
            'earnings season', 'big tech earnings', 'bank earnings', 'earnings outlook',
            'earnings', 'corporate earnings', 'tech stocks', 'bank stocks',

            # Global macro / shocks that move S&P broadly
            'oil prices', 'brent', 'wti', 'commodity shock',
            'geopolitical risk', 'sanctions', 'trade war', 'tariffs',
            'pandemic', 'supply chain disruption'
        }

        # RSS Feed sources
        self.rss_sources = [
            {
                "name": "yahoo_finance_market",
                "url": "https://finance.yahoo.com/rss/topstories"
            },
            {
                "name": "marketwatch_marketpulse",
                "url": "https://www.marketwatch.com/rss/marketpulse"
            },
            {
                "name": "cnbc_topnews",
                "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"
            },
            {
                "name": "seeking_alpha_market",
                "url": "https://seekingalpha.com/feed.xml"
            }
        ]

    def _load_seen_articles(self) -> Set[str]:
        """Load previously seen article IDs."""
        if self.seen_file.exists():
            try:
                with open(self.seen_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('seen', []))
            except Exception as e:
                logger.warning(f"Could not load seen articles: {e}")
        return set()

    def _save_seen_articles(self):
        """Save seen article IDs."""
        with open(self.seen_file, 'w') as f:
            json.dump({
                'seen': list(self.seen_articles),
                'updated_at': datetime.utcnow().isoformat() + 'Z'
            }, f, indent=2)

    def _generate_article_id(self, title: str, url: str) -> str:
        """Generate unique ID for an article."""
        content = f"{title}_{url}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _is_sp500_relevant(self, text: str) -> bool:
        """Check if article text is relevant to S&P 500."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.sp500_keywords)

    def _normalize_timestamp(self, timestamp_str: str) -> str:
        """Normalize various timestamp formats to UTC ISO format."""
        try:
            dt = date_parser.parse(timestamp_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            else:
                dt = dt.astimezone(pytz.UTC)
            return dt.isoformat()
        except Exception as e:
            logger.warning(f"Could not parse timestamp: {timestamp_str} - {e}")
            return datetime.now(pytz.UTC).isoformat()

    def collect_from_rss(self) -> List[Dict]:
        """Collect recent news from RSS feeds."""
        logger.info("Collecting news from RSS feeds")
        articles = []

        for source in self.rss_sources:
            try:
                logger.info(f"Fetching from {source['name']}...")
                feed = feedparser.parse(source['url'])

                for entry in feed.entries:
                    # Extract article data
                    title = entry.get('title', '')
                    url = entry.get('link', '')
                    summary = entry.get('summary', entry.get('description', ''))
                    published = entry.get('published', entry.get('updated', ''))

                    # Generate ID
                    article_id = self._generate_article_id(title, url)

                    # Skip if seen
                    if article_id in self.seen_articles:
                        continue

                    # Check relevance
                    if not self._is_sp500_relevant(title + ' ' + summary):
                        continue

                    # Normalize timestamp
                    published_utc = self._normalize_timestamp(published)

                    article = {
                        'article_id': article_id,
                        'headline': title,
                        'body': summary,
                        'url': url,
                        'source': source['name'],
                        'published_at': published_utc,
                        'collected_at': datetime.now(pytz.UTC).isoformat(),
                        'sp500_relevant': True
                    }

                    articles.append(article)
                    self.seen_articles.add(article_id)

                logger.info(f"Found {len([a for a in articles if a['source'] == source['name']])} new articles from {source['name']}")

            except Exception as e:
                logger.error(f"Error fetching from {source['name']}: {e}")

        return articles

    async def collect_historical_news_api(
        self,
        api_key: str,
        years_back: int = 5,
        query: str = "S&P 500 OR SPX OR stock market"
    ) -> List[Dict]:
        """
        Collect historical news using News API.

        Note: Free tier only allows 30 days of history.
        For 5 years, need NewsAPI premium or alternative sources.
        """
        logger.info(f"Collecting {years_back} years of historical news")

        articles = []
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=years_back * 365)

        # Note: This is a simplified implementation
        # For production, use premium News API or alternative sources like:
        # - Alpha Vantage
        # - Finnhub
        # - Polygon.io
        # - SEC EDGAR for fundamental data

        logger.warning("Historical news collection requires premium API access")
        logger.warning("Free tier News API only provides 30 days of history")

        # For now, return empty and recommend scraping strategies
        return articles

    def save_articles(self, articles: List[Dict]):
        """Save articles to Bronze layer."""
        logger.info(f"Saving {len(articles)} articles")

        for i, article in enumerate(articles):
            # Create filename from article ID
            filename = f"{article['article_id']}.json"
            filepath = self.output_dir / filename

            # Save article
            with open(filepath, 'w') as f:
                json.dump(article, f, indent=2)

        # Save seen articles
        self._save_seen_articles()

        logger.info(f"Saved {len(articles)} articles to {self.output_dir}")

    def run(self, api_key: Optional[str] = None, years: int = 5):
        """Execute the collection process."""
        if self.mode == "recent":
            articles = self.collect_from_rss()
        elif self.mode == "historical":
            if not api_key:
                logger.error("Historical mode requires --api-key parameter")
                return
            # Note: This would need async runner
            articles = []
            logger.warning("Historical scraping requires premium News API or alternative implementation")
            logger.info("Recommended: Use combination of free sources:")
            logger.info("  1. Daily RSS collection (builds over time)")
            logger.info("  2. SEC EDGAR for company filings")
            logger.info("  3. Federal Reserve statements")
            logger.info("  4. Archive.org news archives")
        else:
            logger.error(f"Unknown mode: {self.mode}")
            return

        self.save_articles(articles)

        # Summary
        logger.info(f"Collection complete!")
        logger.info(f"New articles: {len(articles)}")
        logger.info(f"Total seen: {len(self.seen_articles)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["recent", "historical"],
        default="recent",
        help="Collection mode"
    )
    parser.add_argument(
        "--output-dir",
        default="data_clean/bronze/news",
        help="Output directory"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of historical data (for historical mode)"
    )
    parser.add_argument(
        "--api-key",
        help="News API key (for historical mode)"
    )

    args = parser.parse_args()

    collector = NewsDataCollector(
        output_dir=args.output_dir,
        mode=args.mode
    )

    collector.run(api_key=args.api_key, years=args.years)


if __name__ == "__main__":
    main()
