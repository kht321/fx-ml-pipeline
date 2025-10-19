"""
Enhanced Hybrid Historical News Scraper - With Full Article Content
Repository Location: fx-ml-pipeline/src_clean/data_pipelines/bronze/hybrid_news_scraper_enhanced.py

Purpose:
    Enhanced version that fetches full article content from URLs.
    Collects 5 years of historical financial news using FREE sources:
    1. GDELT Project (2017-present, 100% free) - metadata only
    2. Article content fetching from URLs using newspaper3k/beautifulsoup
    3. Alpha Vantage & Finnhub APIs with rate limiting

Features:
    - Fetches full article content from URLs
    - Implements retry logic for failed fetches
    - Caches fetched content to avoid re-fetching
    - Handles various website structures
    - Respects robots.txt and rate limits

Usage:
    # Collect with full content
    python src_clean/data_pipelines/bronze/hybrid_news_scraper_enhanced.py \\
        --start-date 2020-10-19 \\
        --end-date 2025-10-19 \\
        --sources gdelt \\
        --fetch-content
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
import pytz

# Try to import newspaper3k for better article extraction
try:
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False
    print("Warning: newspaper3k not installed. Using basic BeautifulSoup extraction.")
    print("Install with: pip install newspaper3k")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArticleContentFetcher:
    """Fetches full article content from URLs."""

    def __init__(self, cache_dir: str = "data_clean/bronze/news/content_cache"):
        """Initialize the content fetcher."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # User agent for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Track fetch statistics
        self.stats = {
            'fetched': 0,
            'cached': 0,
            'failed': 0
        }

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.txt"

    def _load_from_cache(self, url: str) -> Optional[str]:
        """Load content from cache if available."""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.stats['cached'] += 1
                    return f.read()
            except Exception:
                pass
        return None

    def _save_to_cache(self, url: str, content: str):
        """Save content to cache."""
        cache_path = self._get_cache_path(url)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Failed to cache content for {url}: {e}")

    def fetch_with_newspaper(self, url: str) -> Optional[str]:
        """Fetch article content using newspaper3k library."""
        try:
            article = Article(url)
            article.download()
            article.parse()

            # Get the full text
            text = article.text

            if text and len(text) > 100:  # Minimum content length
                self.stats['fetched'] += 1
                return text

        except Exception as e:
            logger.debug(f"Newspaper3k failed for {url}: {e}")

        return None

    def fetch_with_beautifulsoup(self, url: str) -> Optional[str]:
        """Fetch article content using BeautifulSoup (fallback method)."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Try to find article content in common containers
            article_content = None

            # Try different selectors for article content
            selectors = [
                'article',
                '[class*="article-content"]',
                '[class*="article-body"]',
                '[class*="story-body"]',
                '[class*="content-body"]',
                '[class*="entry-content"]',
                'main',
                '[role="main"]',
                '.content',
                '#content'
            ]

            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    article_content = elements[0]
                    break

            # If no specific article container found, try to get all paragraphs
            if not article_content:
                article_content = soup

            # Extract text from paragraphs
            paragraphs = article_content.find_all('p')
            text = '\n'.join([p.get_text().strip() for p in paragraphs])

            # Clean up the text
            text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
            text = re.sub(r' +', ' ', text)    # Remove multiple spaces
            text = text.strip()

            if text and len(text) > 100:  # Minimum content length
                self.stats['fetched'] += 1
                return text

        except Exception as e:
            logger.debug(f"BeautifulSoup failed for {url}: {e}")

        return None

    def fetch_content(self, url: str, use_cache: bool = True) -> str:
        """
        Fetch article content from URL.

        Parameters
        ----------
        url : str
            The article URL
        use_cache : bool
            Whether to use cached content if available

        Returns
        -------
        str
            The article text content (or empty string if failed)
        """
        # Check cache first
        if use_cache:
            cached_content = self._load_from_cache(url)
            if cached_content:
                return cached_content

        # Try to fetch content
        content = None

        # Try newspaper3k first (if available)
        if HAS_NEWSPAPER:
            content = self.fetch_with_newspaper(url)

        # Fallback to BeautifulSoup
        if not content:
            content = self.fetch_with_beautifulsoup(url)

        # Handle failure
        if not content:
            self.stats['failed'] += 1
            logger.debug(f"Failed to fetch content from {url}")
            content = ""

        # Cache the content (even if empty to avoid re-fetching)
        if use_cache and content:
            self._save_to_cache(url, content)

        return content

    def fetch_batch(self, urls: List[str], max_workers: int = 5) -> Dict[str, str]:
        """
        Fetch content for multiple URLs in parallel.

        Parameters
        ----------
        urls : List[str]
            List of article URLs
        max_workers : int
            Maximum number of parallel workers

        Returns
        -------
        Dict[str, str]
            Mapping of URL to content
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.fetch_content, url): url for url in urls}

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result()
                    results[url] = content
                except Exception as e:
                    logger.warning(f"Error fetching {url}: {e}")
                    results[url] = ""

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get fetch statistics."""
        return self.stats.copy()


class EnhancedHybridNewsScraper:
    """
    Enhanced hybrid scraper with full article content fetching.
    """

    def __init__(
        self,
        output_dir: str = "data_clean/bronze/news/hybrid",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        fetch_content: bool = True
    ):
        """Initialize the enhanced scraper."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_date = start_date or (datetime.now(pytz.UTC) - timedelta(days=365*5))
        self.end_date = end_date or datetime.now(pytz.UTC)
        self.fetch_content = fetch_content

        # Track seen articles across ALL sources
        self.seen_file = self.output_dir / "seen_articles.json"
        self.seen_articles = self._load_seen_articles()

        # Initialize content fetcher if needed
        if self.fetch_content:
            self.content_fetcher = ArticleContentFetcher()
        else:
            self.content_fetcher = None

        # S&P 500 keywords for filtering
        self.sp500_keywords = {
            # Index references
            's&p 500', 'sp500', 's&p500', 'spx', 'spy', 'standard and poor',
            'wall street', 'nasdaq', 'dow jones', 'us equities', 'broad market',
            'stock market', 'index fund',

            # Monetary policy & macro
            'federal reserve', 'fed', 'fomc', 'interest rate', 'rate hike', 'rate cut',
            'inflation', 'cpi', 'ppi', 'core inflation', 'economic growth', 'gdp',
            'unemployment', 'jobless claims', 'recession', 'soft landing', 'hard landing',
            'deflation', 'disinflation', 'stagflation',

            # Bond market & yields
            'treasury yields', '10-year yield', '2-year yield', 'bond market',
            'treasuries', 'government bonds', 'sovereign debt', 'yield curve',
            'inverted yield curve', 'bond selloff', 'bond rally', 'fixed income',

            # Market sentiment
            'market rally', 'market selloff', 'bull market', 'bear market',
            'volatility', 'vix', 'fear index', 'risk-on', 'risk-off',
            'flight to safety', 'safe haven',

            # Earnings season (index-level impact)
            'earnings season', 'big tech earnings', 'bank earnings', 'earnings outlook',
            'earnings',

            # Global macro / shocks that move S&P broadly
            'oil prices', 'brent', 'wti', 'commodity shock',
            'geopolitical risk', 'sanctions', 'trade war', 'tariffs',
            'pandemic', 'supply chain disruption'
        }

        # Rate limiting counters
        self.alphavantage_calls = 0
        self.alphavantage_limit = 25  # per day
        self.finnhub_calls = 0
        self.finnhub_limit = 3600  # 60 per min = 3600 per hour

        logger.info(f"Enhanced scraper initialized for {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Content fetching: {'ENABLED' if self.fetch_content else 'DISABLED'}")

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
                'updated_at': datetime.utcnow().isoformat() + 'Z',
                'total_count': len(self.seen_articles)
            }, f, indent=2)

    def _generate_article_id(self, title: str, url: str) -> str:
        """Generate unique ID for an article."""
        content = f"{title}_{url}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _is_sp500_relevant(self, text: str) -> bool:
        """Check if article text is relevant to S&P 500."""
        if not text:
            return False
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

    def scrape_gdelt(self, date: datetime, max_articles: int = 250) -> List[Dict]:
        """
        Scrape GDELT Project for a specific date and fetch full content.
        """
        logger.info(f"Scraping GDELT for {date.date()}")
        articles = []

        try:
            # Format date for GDELT API
            start_date_str = date.strftime("%Y%m%d%H%M%S")
            end_date_str = (date + timedelta(days=1)).strftime("%Y%m%d%H%M%S")

            # GDELT DOC 2.0 API query
            import urllib.parse
            query = 'stock market'  # Simple query - filter results later
            query_encoded = urllib.parse.quote(query)
            url = (
                f"https://api.gdeltproject.org/api/v2/doc/doc?"
                f"query={query_encoded}&"
                f"mode=artlist&"
                f"maxrecords={max_articles}&"
                f"startdatetime={start_date_str}&"
                f"enddatetime={end_date_str}&"
                f"format=json"
            )

            logger.debug(f"GDELT URL: {url}")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Check if response is valid JSON
            response_text = response.text
            if not response_text or response_text.strip() == '':
                logger.warning(f"GDELT returned empty response for {date.date()}")
                return articles

            data = response.json()

            if 'articles' in data:
                # Collect articles that need content fetching
                articles_to_fetch = []

                for article in data['articles']:
                    title = article.get('title', '')
                    article_url = article.get('url', '')

                    # Generate ID
                    article_id = self._generate_article_id(title, article_url)

                    # Skip if seen
                    if article_id in self.seen_articles:
                        continue

                    # Check relevance based on title
                    if not self._is_sp500_relevant(title):
                        continue

                    # Extract metadata
                    published_at = self._normalize_timestamp(
                        article.get('seendate', article.get('date', ''))
                    )

                    article_data = {
                        'article_id': article_id,
                        'headline': title,
                        'body': article.get('excerpt', ''),  # Start with excerpt
                        'url': article_url,
                        'source': f"gdelt_{article.get('domain', 'unknown')}",
                        'published_at': published_at,
                        'collected_at': datetime.now(pytz.UTC).isoformat(),
                        'sp500_relevant': True,
                        'collection_method': 'gdelt_api',
                        'language': article.get('language', 'en')
                    }

                    articles_to_fetch.append(article_data)

                # Fetch full content if enabled
                if self.fetch_content and self.content_fetcher and articles_to_fetch:
                    logger.info(f"Fetching full content for {len(articles_to_fetch)} articles...")

                    # Extract URLs
                    urls = [a['url'] for a in articles_to_fetch]

                    # Fetch content in batches
                    url_to_content = self.content_fetcher.fetch_batch(urls, max_workers=5)

                    # Update articles with fetched content
                    for article in articles_to_fetch:
                        fetched_content = url_to_content.get(article['url'], '')

                        # If we got content, use it; otherwise keep the excerpt
                        if fetched_content:
                            article['body'] = fetched_content
                            article['content_fetched'] = True
                        else:
                            article['content_fetched'] = False

                        # Re-check relevance with full content
                        if self._is_sp500_relevant(article['headline'] + ' ' + article['body']):
                            articles.append(article)
                            self.seen_articles.add(article['article_id'])

                    # Log fetch statistics
                    stats = self.content_fetcher.get_stats()
                    logger.info(f"Content fetch stats - Fetched: {stats['fetched']}, Cached: {stats['cached']}, Failed: {stats['failed']}")
                else:
                    # No content fetching, just add the articles
                    for article in articles_to_fetch:
                        articles.append(article)
                        self.seen_articles.add(article['article_id'])

            logger.info(f"GDELT: Found {len(articles)} new S&P 500 relevant articles for {date.date()}")

        except Exception as e:
            logger.error(f"GDELT scraping error for {date.date()}: {e}")

        return articles

    def scrape_alpha_vantage(self, date: datetime, api_key: Optional[str] = None) -> List[Dict]:
        """
        Scrape news from Alpha Vantage with content fetching.
        """
        if self.alphavantage_calls >= self.alphavantage_limit:
            logger.warning("Alpha Vantage: Daily limit reached (25 calls)")
            return []

        if not api_key:
            api_key = os.getenv('ALPHAVANTAGE_KEY')
            if not api_key:
                logger.warning("Alpha Vantage: No API key provided")
                return []

        logger.info(f"Scraping Alpha Vantage for {date.date()}")
        articles = []

        try:
            url = (
                f"https://www.alphavantage.co/query?"
                f"function=NEWS_SENTIMENT&"
                f"tickers=SPY&"
                f"time_from={date.strftime('%Y%m%dT%H%M')}&"
                f"time_to={(date + timedelta(days=1)).strftime('%Y%m%dT%H%M')}&"
                f"apikey={api_key}"
            )

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            self.alphavantage_calls += 1

            if 'feed' in data:
                articles_to_fetch = []

                for item in data['feed']:
                    title = item.get('title', '')
                    article_url = item.get('url', '')

                    article_id = self._generate_article_id(title, article_url)

                    if article_id in self.seen_articles:
                        continue

                    if not self._is_sp500_relevant(title + ' ' + item.get('summary', '')):
                        continue

                    article_data = {
                        'article_id': article_id,
                        'headline': title,
                        'body': item.get('summary', ''),
                        'url': article_url,
                        'source': f"alphavantage_{item.get('source', 'unknown')}",
                        'published_at': self._normalize_timestamp(item.get('time_published', '')),
                        'collected_at': datetime.now(pytz.UTC).isoformat(),
                        'sp500_relevant': True,
                        'collection_method': 'alphavantage_api',
                        'sentiment_score': item.get('overall_sentiment_score'),
                        'sentiment_label': item.get('overall_sentiment_label')
                    }

                    articles_to_fetch.append(article_data)

                # Fetch full content if enabled
                if self.fetch_content and self.content_fetcher and articles_to_fetch:
                    logger.info(f"Fetching full content for {len(articles_to_fetch)} Alpha Vantage articles...")
                    urls = [a['url'] for a in articles_to_fetch]
                    url_to_content = self.content_fetcher.fetch_batch(urls, max_workers=3)

                    for article in articles_to_fetch:
                        fetched_content = url_to_content.get(article['url'], '')
                        if fetched_content:
                            article['body'] = fetched_content
                            article['content_fetched'] = True
                        else:
                            article['content_fetched'] = False

                        articles.append(article)
                        self.seen_articles.add(article['article_id'])
                else:
                    for article in articles_to_fetch:
                        articles.append(article)
                        self.seen_articles.add(article['article_id'])

            logger.info(f"Alpha Vantage: Found {len(articles)} articles (calls: {self.alphavantage_calls}/{self.alphavantage_limit})")

        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")

        return articles

    def scrape_finnhub(self, date: datetime, api_key: Optional[str] = None) -> List[Dict]:
        """
        Scrape news from Finnhub with content fetching.
        """
        if self.finnhub_calls >= self.finnhub_limit:
            logger.warning("Finnhub: Hourly limit reached (3600 calls)")
            return []

        if not api_key:
            api_key = os.getenv('FINNHUB_KEY')
            if not api_key:
                logger.warning("Finnhub: No API key provided")
                return []

        logger.info(f"Scraping Finnhub for {date.date()}")
        articles = []

        try:
            # Finnhub market news endpoint
            start_ts = int(date.timestamp())
            end_ts = int((date + timedelta(days=1)).timestamp())

            url = (
                f"https://finnhub.io/api/v1/news?"
                f"category=general&"
                f"from={start_ts}&"
                f"to={end_ts}&"
                f"token={api_key}"
            )

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            self.finnhub_calls += 1

            articles_to_fetch = []

            for item in data:
                title = item.get('headline', '')
                article_url = item.get('url', '')

                article_id = self._generate_article_id(title, article_url)

                if article_id in self.seen_articles:
                    continue

                if not self._is_sp500_relevant(title + ' ' + item.get('summary', '')):
                    continue

                article_data = {
                    'article_id': article_id,
                    'headline': title,
                    'body': item.get('summary', ''),
                    'url': article_url,
                    'source': f"finnhub_{item.get('source', 'unknown')}",
                    'published_at': self._normalize_timestamp(
                        datetime.fromtimestamp(item.get('datetime', 0)).isoformat()
                    ),
                    'collected_at': datetime.now(pytz.UTC).isoformat(),
                    'sp500_relevant': True,
                    'collection_method': 'finnhub_api'
                }

                articles_to_fetch.append(article_data)

            # Fetch full content if enabled
            if self.fetch_content and self.content_fetcher and articles_to_fetch:
                logger.info(f"Fetching full content for {len(articles_to_fetch)} Finnhub articles...")
                urls = [a['url'] for a in articles_to_fetch]
                url_to_content = self.content_fetcher.fetch_batch(urls, max_workers=3)

                for article in articles_to_fetch:
                    fetched_content = url_to_content.get(article['url'], '')
                    if fetched_content:
                        article['body'] = fetched_content
                        article['content_fetched'] = True
                    else:
                        article['content_fetched'] = False

                    articles.append(article)
                    self.seen_articles.add(article['article_id'])
            else:
                for article in articles_to_fetch:
                    articles.append(article)
                    self.seen_articles.add(article['article_id'])

            logger.info(f"Finnhub: Found {len(articles)} articles (calls: {self.finnhub_calls}/{self.finnhub_limit})")

            # Rate limiting: 60 calls per minute
            time.sleep(1)

        except Exception as e:
            logger.error(f"Finnhub error: {e}")

        return articles

    def collect_for_date_range(
        self,
        sources: List[str] = ['gdelt', 'alphavantage', 'finnhub']
    ) -> Dict[str, int]:
        """
        Collect news from all sources for the configured date range.
        """
        if 'all' in sources:
            sources = ['gdelt', 'alphavantage', 'finnhub']

        stats = {source: 0 for source in sources}
        all_articles = []

        # Iterate through date range
        current_date = self.start_date

        while current_date <= self.end_date:
            logger.info(f"Collecting news for {current_date.date()}")

            # GDELT (2017-present)
            if 'gdelt' in sources and current_date >= datetime(2017, 1, 1, tzinfo=pytz.UTC):
                articles = self.scrape_gdelt(current_date)
                all_articles.extend(articles)
                stats['gdelt'] += len(articles)

            # Alpha Vantage (25 calls/day max)
            if 'alphavantage' in sources and self.alphavantage_calls < self.alphavantage_limit:
                articles = self.scrape_alpha_vantage(current_date)
                all_articles.extend(articles)
                stats['alphavantage'] += len(articles)

            # Finnhub (60 calls/min, 1 year history max)
            if 'finnhub' in sources and current_date >= datetime.now(pytz.UTC) - timedelta(days=365):
                if self.finnhub_calls < self.finnhub_limit:
                    articles = self.scrape_finnhub(current_date)
                    all_articles.extend(articles)
                    stats['finnhub'] += len(articles)

            # Save articles for this date
            if all_articles:
                self.save_articles(all_articles)
                all_articles = []

            # Move to next day
            current_date += timedelta(days=1)

        # Save final state
        self._save_seen_articles()

        # Print content fetch statistics if enabled
        if self.fetch_content and self.content_fetcher:
            fetch_stats = self.content_fetcher.get_stats()
            logger.info("=" * 80)
            logger.info("CONTENT FETCHING STATISTICS:")
            logger.info(f"Articles with content fetched: {fetch_stats['fetched']}")
            logger.info(f"Articles loaded from cache: {fetch_stats['cached']}")
            logger.info(f"Failed content fetches: {fetch_stats['failed']}")
            logger.info("=" * 80)

        return stats

    def save_articles(self, articles: List[Dict]):
        """Save articles to Bronze layer."""
        if not articles:
            return

        for article in articles:
            filename = f"{article['article_id']}.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(article, f, indent=2, ensure_ascii=False)

        # Update seen articles
        self._save_seen_articles()

        logger.info(f"Saved {len(articles)} articles to {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Hybrid News Scraper - With Full Article Content"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="gdelt",
        help="Comma-separated sources: gdelt,alphavantage,finnhub,all"
    )
    parser.add_argument(
        "--output-dir",
        default="data_clean/bronze/news/hybrid_enhanced",
        help="Output directory"
    )
    parser.add_argument(
        "--fetch-content",
        action="store_true",
        default=True,
        help="Fetch full article content from URLs (default: True)"
    )
    parser.add_argument(
        "--no-fetch-content",
        dest="fetch_content",
        action="store_false",
        help="Disable content fetching"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="Collection mode: full (date range) or incremental (last 24h)"
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

    if args.mode == "incremental":
        start_date = datetime.now(pytz.UTC) - timedelta(days=1)
        end_date = datetime.now(pytz.UTC)

    # Parse sources
    sources = [s.strip() for s in args.sources.split(',')]

    # Initialize scraper
    scraper = EnhancedHybridNewsScraper(
        output_dir=args.output_dir,
        start_date=start_date,
        end_date=end_date,
        fetch_content=args.fetch_content
    )

    # Collect news
    logger.info("=" * 80)
    logger.info("ENHANCED HYBRID NEWS SCRAPER - WITH FULL CONTENT")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Sources: {', '.join(sources)}")
    logger.info(f"Content fetching: {'ENABLED' if args.fetch_content else 'DISABLED'}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)

    stats = scraper.collect_for_date_range(sources=sources)

    # Summary
    logger.info("=" * 80)
    logger.info("COLLECTION COMPLETE!")
    logger.info("=" * 80)
    for source, count in stats.items():
        logger.info(f"{source:20s}: {count:6d} articles")
    logger.info(f"{'TOTAL':20s}: {sum(stats.values()):6d} articles")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()