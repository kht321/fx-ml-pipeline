#!/usr/bin/env python3
"""
Historical S&P 500 News Scraper

Scrapes 5 years of historical S&P 500-related news articles to match
the market data period (Oct 2020 - Oct 2025).

Uses multiple sources:
1. News API (free tier: 1 month historical)
2. Alpha Vantage News API (free tier available)
3. Finnhub News API (free tier available)
4. MarketWatch RSS archives
5. Web scraping for older articles

For production: Consider paid APIs for full historical coverage.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API keys from .env
load_dotenv()


class HistoricalNewsScraper:
    """Scrapes historical S&P 500 news from multiple sources."""

    def __init__(self, output_dir: str = "data/news/bronze/raw_articles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # API keys from environment
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        self.alphavantage_key = os.getenv("ALPHAVANTAGE_KEY", "")
        self.finnhub_key = os.getenv("FINNHUB_KEY", "")

        # S&P 500 relevant keywords
        self.keywords = [
            "S&P 500", "SPX", "stock market", "stocks",
            "equity market", "Wall Street", "NASDAQ",
            "Dow Jones", "market rally", "market selloff"
        ]

        # Track scraped articles
        self.articles_scraped = 0
        self.articles_saved = 0

    def scrape_newsapi(self, start_date: datetime, end_date: datetime, keyword: str) -> List[Dict]:
        """
        Scrape from News API.

        Note: Free tier only provides 1 month of historical data.
        For full 5 years, need paid plan (~$449/month for Business tier).
        """
        if not self.newsapi_key:
            logger.warning("NewsAPI key not found. Set NEWSAPI_KEY in .env")
            return []

        articles = []
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": keyword,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": self.newsapi_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "ok":
                for article in data.get("articles", []):
                    articles.append({
                        "article_id": f"newsapi_{hash(article.get('url', ''))}",
                        "published_at": article.get("publishedAt"),
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "headline": article.get("title", ""),
                        "content": article.get("content", "") or article.get("description", ""),
                        "author": article.get("author"),
                        "url": article.get("url"),
                        "category": "markets",
                        "tags": [keyword],
                        "scraped_at": datetime.utcnow().isoformat() + "Z",
                        "api_source": "newsapi"
                    })

            logger.info(f"NewsAPI: Found {len(articles)} articles for '{keyword}'")
            time.sleep(1)  # Rate limiting

        except requests.exceptions.HTTPError as e:
            if "426" in str(e):
                logger.warning("NewsAPI: Free tier only provides 1 month history. Consider upgrading.")
            else:
                logger.error(f"NewsAPI error: {e}")
        except Exception as e:
            logger.error(f"NewsAPI exception: {e}")

        return articles

    def scrape_alphavantage(self, ticker: str = "SPX") -> List[Dict]:
        """
        Scrape from Alpha Vantage News & Sentiment API.

        Free tier: 25 requests/day. Provides recent news with sentiment.
        """
        if not self.alphavantage_key:
            logger.warning("Alpha Vantage key not found. Set ALPHAVANTAGE_KEY in .env")
            return []

        articles = []
        url = "https://www.alphavantage.co/query"

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": self.alphavantage_key,
            "limit": 1000  # Max results
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for item in data.get("feed", []):
                # Alpha Vantage provides sentiment scores too!
                ticker_sentiment = {}
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker") == ticker:
                        ticker_sentiment = ts

                articles.append({
                    "article_id": f"alphavantage_{hash(item.get('url', ''))}",
                    "published_at": item.get("time_published"),
                    "source": item.get("source", "Unknown"),
                    "headline": item.get("title", ""),
                    "content": item.get("summary", ""),
                    "author": ",".join(item.get("authors", [])),
                    "url": item.get("url"),
                    "category": item.get("category_within_source", "markets"),
                    "tags": [ticker] + item.get("topics", []),
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                    "api_source": "alphavantage",
                    "sentiment_score": ticker_sentiment.get("ticker_sentiment_score"),
                    "sentiment_label": ticker_sentiment.get("ticker_sentiment_label")
                })

            logger.info(f"Alpha Vantage: Found {len(articles)} articles")
            time.sleep(12)  # Rate limiting: 5 calls/minute

        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")

        return articles

    def scrape_finnhub(self, category: str = "general") -> List[Dict]:
        """
        Scrape from Finnhub Market News API.

        Free tier: 60 calls/minute. Recent news only.
        """
        if not self.finnhub_key:
            logger.warning("Finnhub key not found. Set FINNHUB_KEY in .env")
            return []

        articles = []
        url = "https://finnhub.io/api/v1/news"

        params = {
            "category": category,
            "token": self.finnhub_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for item in data:
                articles.append({
                    "article_id": f"finnhub_{item.get('id', hash(item.get('url', '')))}",
                    "published_at": datetime.fromtimestamp(item.get("datetime", 0)).isoformat() + "Z",
                    "source": item.get("source", "Unknown"),
                    "headline": item.get("headline", ""),
                    "content": item.get("summary", ""),
                    "author": None,
                    "url": item.get("url"),
                    "category": category,
                    "tags": ["SPX", "market news"],
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                    "api_source": "finnhub",
                    "related_symbols": item.get("related", "").split(",")
                })

            logger.info(f"Finnhub: Found {len(articles)} articles")
            time.sleep(1)  # Rate limiting

        except Exception as e:
            logger.error(f"Finnhub error: {e}")

        return articles

    def scrape_free_sources(self) -> List[Dict]:
        """
        Scrape from free sources that don't require API keys.

        These provide recent news but are good for building up a dataset.
        """
        articles = []

        # Example: Using free RSS feeds or public APIs
        # This is a placeholder for demonstration

        logger.info("Free sources: Implement RSS parsing or public APIs here")

        return articles

    def save_articles(self, articles: List[Dict], date_str: str = None):
        """Save articles to NDJSON file."""
        if not articles:
            return

        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        output_file = self.output_dir / f"sp500_news_{date_str}.ndjson"

        with open(output_file, 'a') as f:
            for article in articles:
                f.write(json.dumps(article) + '\n')
                self.articles_saved += 1

        logger.info(f"Saved {len(articles)} articles to {output_file}")

    def scrape_date_range(self, start_date: datetime, end_date: datetime):
        """
        Scrape news for a date range.

        Note: Due to API limitations, this will mainly get recent data.
        For full 5-year history, consider:
        1. Paid API plans
        2. Web scraping news archives
        3. Purchasing historical news datasets
        """
        logger.info(f"Scraping news from {start_date} to {end_date}")

        # Chunk into smaller periods (e.g., monthly)
        current_date = start_date
        total_articles = []

        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=30), end_date)

            logger.info(f"Processing period: {current_date.date()} to {chunk_end.date()}")

            # Try each source
            for keyword in self.keywords[:2]:  # Limit keywords to avoid rate limits
                # NewsAPI (limited to recent month on free tier)
                articles = self.scrape_newsapi(current_date, chunk_end, keyword)
                total_articles.extend(articles)
                self.articles_scraped += len(articles)

            # Save periodically
            if total_articles:
                self.save_articles(
                    total_articles,
                    date_str=current_date.strftime("%Y%m")
                )
                total_articles = []

            current_date = chunk_end

    def scrape_recent_with_all_sources(self):
        """
        Scrape recent news using all available sources.

        This is more practical for free tiers which focus on recent data.
        """
        logger.info("Scraping recent news from all sources...")

        all_articles = []

        # Alpha Vantage
        articles = self.scrape_alphavantage("SPX")
        all_articles.extend(articles)

        # Finnhub
        articles = self.scrape_finnhub("general")
        all_articles.extend(articles)

        # NewsAPI (last month)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        for keyword in ["S&P 500", "stock market"]:
            articles = self.scrape_newsapi(start_date, end_date, keyword)
            all_articles.extend(articles)

        # Save all
        self.save_articles(all_articles)

        logger.info(f"Total articles scraped: {len(all_articles)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-10-13",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-10-10",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/news/bronze/raw_articles",
        help="Output directory"
    )
    parser.add_argument(
        "--recent-only",
        action="store_true",
        help="Only scrape recent news (recommended for free tiers)"
    )

    args = parser.parse_args()

    scraper = HistoricalNewsScraper(output_dir=args.output_dir)

    logger.info("=" * 80)
    logger.info("S&P 500 Historical News Scraper")
    logger.info("=" * 80)
    logger.info("")

    # Check API keys
    has_keys = False
    if scraper.newsapi_key:
        logger.info("✓ NewsAPI key found")
        has_keys = True
    else:
        logger.warning("✗ NewsAPI key not found (set NEWSAPI_KEY in .env)")

    if scraper.alphavantage_key:
        logger.info("✓ Alpha Vantage key found")
        has_keys = True
    else:
        logger.warning("✗ Alpha Vantage key not found (set ALPHAVANTAGE_KEY in .env)")

    if scraper.finnhub_key:
        logger.info("✓ Finnhub key found")
        has_keys = True
    else:
        logger.warning("✗ Finnhub key not found (set FINNHUB_KEY in .env)")

    if not has_keys:
        logger.error("")
        logger.error("⚠️  No API keys found!")
        logger.error("")
        logger.error("To scrape news, you need at least one API key.")
        logger.error("Add to your .env file:")
        logger.error("")
        logger.error("  NEWSAPI_KEY=your_key_here          # Get from: https://newsapi.org/")
        logger.error("  ALPHAVANTAGE_KEY=your_key_here     # Get from: https://www.alphavantage.co/")
        logger.error("  FINNHUB_KEY=your_key_here          # Get from: https://finnhub.io/")
        logger.error("")
        logger.error("All offer free tiers!")
        return

    logger.info("")
    logger.info("⚠️  IMPORTANT: Free API tiers have limitations:")
    logger.info("  - NewsAPI: Only 1 month historical (need paid for 5 years)")
    logger.info("  - Alpha Vantage: 25 requests/day")
    logger.info("  - Finnhub: Recent news only")
    logger.info("")
    logger.info("For full 5-year historical news, consider:")
    logger.info("  1. NewsAPI Business plan ($449/month)")
    logger.info("  2. Purchasing historical datasets")
    logger.info("  3. Running this script daily to build up history")
    logger.info("")

    if args.recent_only:
        logger.info("Mode: Recent news only (recommended for free tiers)")
        scraper.scrape_recent_with_all_sources()
    else:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        logger.info(f"Mode: Historical scraping {start_date.date()} to {end_date.date()}")
        logger.info("Note: Free tiers will only provide recent data")
        scraper.scrape_date_range(start_date, end_date)

    logger.info("")
    logger.info("=" * 80)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Articles scraped: {scraper.articles_scraped}")
    logger.info(f"Articles saved: {scraper.articles_saved}")
    logger.info(f"Output directory: {scraper.output_dir}")


if __name__ == "__main__":
    main()
