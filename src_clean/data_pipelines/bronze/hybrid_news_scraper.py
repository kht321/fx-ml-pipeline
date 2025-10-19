"""
Enhanced Hybrid News Scraper with 429 Error Handling
Repository Location: fx-ml-pipeline/src_clean/data_pipelines/bronze/hybrid_news_scraper.py

Purpose:
    Production news scraper with robust 429 error handling and rate limiting.
    Collects 5+ years of S&P 500 relevant news with full article content.

Key Features:
    - Exponential backoff for 429 errors
    - Per-domain rate limiting
    - Rotating user agents
    - Request throttling
    - Proxy support (optional)
    - Smart retry logic
    - Resume capability after failures
    - Full article content fetching
    - 7-day content caching

Usage:
    # Collect 5 years of news (conservative)
    python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
        --start-date 2020-10-19 \
        --end-date 2025-10-19 \
        --sources gdelt \
        --fetch-content \
        --max-workers 1 \
        --delay-between-requests 2.0

    # Daily incremental collection
    python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
        --mode incremental \
        --sources all \
        --fetch-content
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Per-domain rate limiter to prevent 429 errors."""

    def __init__(self, default_delay: float = 1.0):
        """
        Initialize rate limiter.

        Parameters
        ----------
        default_delay : float
            Default delay between requests to the same domain (seconds)
        """
        self.default_delay = default_delay
        self.domain_last_request = defaultdict(float)
        self.domain_delays = defaultdict(lambda: self.default_delay)
        self.lock = Lock()

        # Specific domain delays (customize based on experience)
        self.domain_delays.update({
            'reuters.com': 2.0,
            'bloomberg.com': 3.0,
            'wsj.com': 2.5,
            'ft.com': 2.0,
            'cnbc.com': 1.5,
            'marketwatch.com': 1.5,
            'yahoo.com': 1.0,
            'benzinga.com': 1.0,
            'seekingalpha.com': 2.0
        })

    def wait_if_needed(self, url: str):
        """Wait if necessary before making request to domain."""
        domain = urlparse(url).netloc

        with self.lock:
            last_request = self.domain_last_request[domain]
            delay = self.domain_delays[domain]

            if last_request > 0:
                elapsed = time.time() - last_request
                if elapsed < delay:
                    wait_time = delay - elapsed
                    logger.debug(f"Rate limiting {domain}: waiting {wait_time:.2f}s")
                    time.sleep(wait_time)

            self.domain_last_request[domain] = time.time()

    def increase_delay(self, url: str):
        """Increase delay for a domain after 429 error."""
        domain = urlparse(url).netloc

        with self.lock:
            current_delay = self.domain_delays[domain]
            new_delay = min(current_delay * 2, 60)  # Max 60 seconds
            self.domain_delays[domain] = new_delay
            logger.warning(f"Increased delay for {domain}: {current_delay:.1f}s -> {new_delay:.1f}s")

    def reset_delay(self, url: str):
        """Reset delay for a domain after successful request."""
        domain = urlparse(url).netloc

        with self.lock:
            # Gradually decrease delay on success
            current_delay = self.domain_delays[domain]
            if current_delay > self.default_delay:
                new_delay = max(current_delay * 0.9, self.default_delay)
                self.domain_delays[domain] = new_delay


class UserAgentRotator:
    """Rotate user agents to avoid detection."""

    def __init__(self):
        self.user_agents = [
            # Chrome on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',

            # Chrome on Mac
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',

            # Firefox on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0',

            # Safari on Mac
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15',

            # Edge on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.46'
        ]

    def get_random_agent(self) -> str:
        """Get a random user agent."""
        return random.choice(self.user_agents)


class EnhancedArticleContentFetcher:
    """Fetches article content with robust 429 error handling."""

    def __init__(
        self,
        cache_dir: str = "data_clean/bronze/news/content_cache",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        use_proxy: bool = False,
        proxy_list: Optional[List[str]] = None
    ):
        """
        Initialize content fetcher with rate limiting.

        Parameters
        ----------
        cache_dir : str
            Directory for caching content
        max_retries : int
            Maximum number of retries for 429 errors
        initial_backoff : float
            Initial backoff time in seconds
        max_backoff : float
            Maximum backoff time in seconds
        use_proxy : bool
            Whether to use proxy rotation
        proxy_list : List[str]
            List of proxy URLs (e.g., ['http://proxy1:8080', 'http://proxy2:8080'])
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.rate_limiter = RateLimiter()
        self.user_agent_rotator = UserAgentRotator()

        # Retry configuration
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

        # Proxy configuration
        self.use_proxy = use_proxy
        self.proxy_list = proxy_list or []
        self.current_proxy_index = 0

        # Create session with retry strategy
        self.session = self._create_session()

        # Statistics
        self.stats = {
            'fetched': 0,
            'cached': 0,
            'failed': 0,
            'rate_limited': 0,
            'retries': 0
        }

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,  # Initial retries for connection errors
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],  # Not 429 - we handle that separately
            allowed_methods=["GET"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_proxy(self) -> Optional[Dict[str, str]]:
        """Get next proxy from rotation."""
        if not self.use_proxy or not self.proxy_list:
            return None

        proxy_url = self.proxy_list[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)

        return {
            'http': proxy_url,
            'https': proxy_url
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
                # Check if cache is recent (within 7 days)
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age < 7 * 24 * 3600:  # 7 days
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

    def fetch_with_retry(self, url: str, method: str = 'beautifulsoup') -> Optional[str]:
        """
        Fetch content with exponential backoff for 429 errors.

        Parameters
        ----------
        url : str
            URL to fetch
        method : str
            'newspaper' or 'beautifulsoup'

        Returns
        -------
        Optional[str]
            Article content or None if failed
        """
        backoff = self.initial_backoff

        for attempt in range(self.max_retries):
            try:
                # Wait based on rate limiter
                self.rate_limiter.wait_if_needed(url)

                # Get headers with rotating user agent
                headers = {
                    'User-Agent': self.user_agent_rotator.get_random_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }

                # Get proxy if enabled
                proxies = self._get_proxy()

                # Make request
                response = self.session.get(
                    url,
                    headers=headers,
                    proxies=proxies,
                    timeout=10,
                    allow_redirects=True
                )

                # Check for rate limiting
                if response.status_code == 429:
                    self.stats['rate_limited'] += 1
                    self.stats['retries'] += 1

                    # Check for Retry-After header
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            wait_time = int(retry_after)
                        except ValueError:
                            # Might be a date
                            wait_time = backoff
                    else:
                        wait_time = backoff

                    logger.warning(f"Rate limited (429) for {urlparse(url).netloc}. "
                                 f"Attempt {attempt + 1}/{self.max_retries}. "
                                 f"Waiting {wait_time:.1f}s...")

                    # Increase delay for this domain
                    self.rate_limiter.increase_delay(url)

                    # Wait with exponential backoff
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, self.max_backoff)

                    continue

                # Check for success
                response.raise_for_status()

                # Reset delay on success
                self.rate_limiter.reset_delay(url)

                # Extract content based on method
                if method == 'newspaper' and HAS_NEWSPAPER:
                    content = self._extract_with_newspaper(url, response.text)
                else:
                    content = self._extract_with_beautifulsoup(response.text)

                if content and len(content) > 100:
                    self.stats['fetched'] += 1
                    return content

                return None

            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code != 429:
                    # Not a rate limit error, don't retry
                    logger.debug(f"HTTP error for {url}: {e}")
                    self.stats['failed'] += 1
                    return None

            except Exception as e:
                logger.debug(f"Error fetching {url}: {e}")
                if attempt == self.max_retries - 1:
                    self.stats['failed'] += 1
                    return None

                # Wait before retry
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)

        self.stats['failed'] += 1
        return None

    def _extract_with_newspaper(self, url: str, html: str) -> Optional[str]:
        """Extract content using newspaper3k."""
        try:
            article = Article(url)
            article.set_html(html)
            article.parse()
            return article.text
        except Exception as e:
            logger.debug(f"Newspaper extraction failed: {e}")
            return None

    def _extract_with_beautifulsoup(self, html: str) -> Optional[str]:
        """Extract content using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()

            # Try different content selectors
            content_selectors = [
                'article',
                '[class*="article-content"]',
                '[class*="article-body"]',
                '[class*="story-body"]',
                '[class*="content-body"]',
                '[class*="entry-content"]',
                '[itemprop="articleBody"]',
                'main',
                '[role="main"]',
                '.content',
                '#content'
            ]

            article_content = None
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    article_content = elements[0]
                    break

            if not article_content:
                article_content = soup.body or soup

            # Extract paragraphs
            paragraphs = article_content.find_all('p')
            text = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])

            # Clean up
            text = ' '.join(text.split())  # Normalize whitespace

            return text if len(text) > 100 else None

        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {e}")
            return None

    def fetch_content(self, url: str, use_cache: bool = True) -> str:
        """
        Fetch article content with caching and retry logic.

        Parameters
        ----------
        url : str
            Article URL
        use_cache : bool
            Whether to use cached content

        Returns
        -------
        str
            Article content (empty string if failed)
        """
        # Check cache first
        if use_cache:
            cached_content = self._load_from_cache(url)
            if cached_content:
                return cached_content

        # Try to fetch with retry logic
        content = self.fetch_with_retry(url)

        if not content:
            content = ""

        # Cache the result (even if empty to avoid re-fetching)
        if use_cache and content:
            self._save_to_cache(url, content)

        return content

    def fetch_batch(
        self,
        urls: List[str],
        max_workers: int = 2,
        delay_between_batches: float = 1.0
    ) -> Dict[str, str]:
        """
        Fetch content for multiple URLs with controlled concurrency.

        Parameters
        ----------
        urls : List[str]
            List of URLs to fetch
        max_workers : int
            Maximum concurrent workers (keep low to avoid 429)
        delay_between_batches : float
            Delay between batches of requests

        Returns
        -------
        Dict[str, str]
            Mapping of URL to content
        """
        results = {}

        # Group URLs by domain to better manage rate limiting
        domain_urls = defaultdict(list)
        for url in urls:
            domain = urlparse(url).netloc
            domain_urls[domain].append(url)

        # Process domains sequentially, URLs within domain in parallel
        for domain, domain_url_list in domain_urls.items():
            logger.info(f"Fetching {len(domain_url_list)} articles from {domain}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {
                    executor.submit(self.fetch_content, url): url
                    for url in domain_url_list
                }

                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        results[url] = content
                    except Exception as e:
                        logger.warning(f"Error fetching {url}: {e}")
                        results[url] = ""

            # Delay between domains
            if delay_between_batches > 0:
                time.sleep(delay_between_batches)

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get fetch statistics."""
        return self.stats.copy()


class RateLimitedHybridNewsScraper:
    """
    Hybrid scraper with enhanced 429 error handling.
    """

    def __init__(
        self,
        output_dir: str = "data_clean/bronze/news/hybrid",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        fetch_content: bool = True,
        max_workers: int = 2,
        delay_between_requests: float = 1.0
    ):
        """Initialize the rate-limited scraper."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_date = start_date or (datetime.now(pytz.UTC) - timedelta(days=365*5))
        self.end_date = end_date or datetime.now(pytz.UTC)
        self.fetch_content = fetch_content
        self.max_workers = max_workers
        self.delay_between_requests = delay_between_requests

        # Track seen articles
        self.seen_file = self.output_dir / "seen_articles.json"
        self.seen_articles = self._load_seen_articles()

        # Initialize content fetcher with rate limiting
        if self.fetch_content:
            self.content_fetcher = EnhancedArticleContentFetcher(
                max_retries=5,
                initial_backoff=2.0,
                max_backoff=60.0
            )
        else:
            self.content_fetcher = None

        # S&P 500 keywords
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

            # Earnings season
            'earnings season', 'big tech earnings', 'bank earnings', 'earnings outlook',
            'earnings',

            # Global macro / shocks
            'oil prices', 'brent', 'wti', 'commodity shock',
            'geopolitical risk', 'sanctions', 'trade war', 'tariffs',
            'pandemic', 'supply chain disruption'
        }

        # API rate limits
        self.alphavantage_calls = 0
        self.alphavantage_limit = 25
        self.finnhub_calls = 0
        self.finnhub_limit = 3600

        logger.info(f"Rate-limited scraper initialized")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Max workers: {max_workers}, Delay: {delay_between_requests}s")

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
        """Check if article is S&P 500 relevant."""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.sp500_keywords)

    def _normalize_timestamp(self, timestamp_str: str) -> str:
        """Normalize timestamp to UTC ISO format."""
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
        """Scrape GDELT with rate-limited content fetching."""
        logger.info(f"Scraping GDELT for {date.date()}")
        articles = []

        try:
            # GDELT API call (no rate limiting needed - it's free)
            start_date_str = date.strftime("%Y%m%d%H%M%S")
            end_date_str = (date + timedelta(days=1)).strftime("%Y%m%d%H%M%S")

            import urllib.parse
            query = 'stock market'
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

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            response_text = response.text
            if not response_text or response_text.strip() == '':
                logger.warning(f"GDELT returned empty response for {date.date()}")
                return articles

            data = response.json()

            if 'articles' in data:
                articles_to_fetch = []

                for article in data['articles']:
                    title = article.get('title', '')
                    article_url = article.get('url', '')

                    article_id = self._generate_article_id(title, article_url)

                    if article_id in self.seen_articles:
                        continue

                    if not self._is_sp500_relevant(title):
                        continue

                    published_at = self._normalize_timestamp(
                        article.get('seendate', article.get('date', ''))
                    )

                    article_data = {
                        'article_id': article_id,
                        'headline': title,
                        'body': article.get('excerpt', ''),
                        'url': article_url,
                        'source': f"gdelt_{article.get('domain', 'unknown')}",
                        'published_at': published_at,
                        'collected_at': datetime.now(pytz.UTC).isoformat(),
                        'sp500_relevant': True,
                        'collection_method': 'gdelt_api',
                        'language': article.get('language', 'en')
                    }

                    articles_to_fetch.append(article_data)

                # Fetch content with rate limiting
                if self.fetch_content and self.content_fetcher and articles_to_fetch:
                    logger.info(f"Fetching content for {len(articles_to_fetch)} articles "
                              f"(max {self.max_workers} workers, {self.delay_between_requests}s delay)")

                    urls = [a['url'] for a in articles_to_fetch]

                    # Fetch with controlled concurrency and delays
                    url_to_content = self.content_fetcher.fetch_batch(
                        urls,
                        max_workers=self.max_workers,
                        delay_between_batches=self.delay_between_requests
                    )

                    for article in articles_to_fetch:
                        fetched_content = url_to_content.get(article['url'], '')

                        if fetched_content:
                            article['body'] = fetched_content
                            article['content_fetched'] = True
                        else:
                            article['content_fetched'] = False

                        if self._is_sp500_relevant(article['headline'] + ' ' + article['body']):
                            articles.append(article)
                            self.seen_articles.add(article['article_id'])

                    # Log statistics
                    stats = self.content_fetcher.get_stats()
                    logger.info(f"Fetch stats - Success: {stats['fetched']}, "
                              f"Cached: {stats['cached']}, Failed: {stats['failed']}, "
                              f"Rate limited: {stats['rate_limited']}, Retries: {stats['retries']}")
                else:
                    for article in articles_to_fetch:
                        articles.append(article)
                        self.seen_articles.add(article['article_id'])

            logger.info(f"GDELT: Collected {len(articles)} S&P 500 relevant articles for {date.date()}")

        except Exception as e:
            logger.error(f"GDELT scraping error for {date.date()}: {e}")

        return articles

    def collect_for_date_range(
        self,
        sources: List[str] = ['gdelt']
    ) -> Dict[str, int]:
        """Collect news for date range with rate limiting."""
        if 'all' in sources:
            sources = ['gdelt', 'alphavantage', 'finnhub']

        stats = {source: 0 for source in sources}
        all_articles = []

        current_date = self.start_date

        while current_date <= self.end_date:
            logger.info(f"Collecting news for {current_date.date()}")

            # GDELT (2017-present)
            if 'gdelt' in sources and current_date >= datetime(2017, 1, 1, tzinfo=pytz.UTC):
                articles = self.scrape_gdelt(current_date)
                all_articles.extend(articles)
                stats['gdelt'] += len(articles)

            # Save articles
            if all_articles:
                self.save_articles(all_articles)
                all_articles = []

            # Move to next day
            current_date += timedelta(days=1)

            # Optional: Add delay between days to be extra careful
            if self.delay_between_requests > 0:
                time.sleep(self.delay_between_requests)

        # Save final state
        self._save_seen_articles()

        # Print final statistics
        if self.fetch_content and self.content_fetcher:
            fetch_stats = self.content_fetcher.get_stats()
            logger.info("=" * 80)
            logger.info("CONTENT FETCHING FINAL STATISTICS:")
            logger.info(f"Successfully fetched: {fetch_stats['fetched']}")
            logger.info(f"Loaded from cache: {fetch_stats['cached']}")
            logger.info(f"Failed to fetch: {fetch_stats['failed']}")
            logger.info(f"Rate limited (429): {fetch_stats['rate_limited']}")
            logger.info(f"Total retries: {fetch_stats['retries']}")
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

        self._save_seen_articles()
        logger.info(f"Saved {len(articles)} articles to {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rate-Limited Hybrid News Scraper - Handles 429 Errors"
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
        default="data_clean/bronze/news/hybrid_rate_limited",
        help="Output directory"
    )
    parser.add_argument(
        "--fetch-content",
        action="store_true",
        default=True,
        help="Fetch full article content from URLs"
    )
    parser.add_argument(
        "--no-fetch-content",
        dest="fetch_content",
        action="store_false",
        help="Disable content fetching"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum concurrent workers for content fetching (default: 2, keep low to avoid 429)"
    )
    parser.add_argument(
        "--delay-between-requests",
        type=float,
        default=1.0,
        help="Delay between request batches in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="Collection mode"
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
    scraper = RateLimitedHybridNewsScraper(
        output_dir=args.output_dir,
        start_date=start_date,
        end_date=end_date,
        fetch_content=args.fetch_content,
        max_workers=args.max_workers,
        delay_between_requests=args.delay_between_requests
    )

    # Collect news
    logger.info("=" * 80)
    logger.info("RATE-LIMITED HYBRID NEWS SCRAPER")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Sources: {', '.join(sources)}")
    logger.info(f"Content fetching: {'ENABLED' if args.fetch_content else 'DISABLED'}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Request delay: {args.delay_between_requests}s")
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