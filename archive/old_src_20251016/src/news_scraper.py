"""Real-time news scraping pipeline for SGD-relevant financial news.

Scrapes live financial news from major sources and saves to Bronze layer
for training data collection throughout 2025.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsSource:
    """Configuration for a news source."""

    def __init__(self, name: str, rss_url: str, base_url: str, selectors: Dict[str, str]):
        self.name = name
        self.rss_url = rss_url
        self.base_url = base_url
        self.selectors = selectors  # CSS selectors for content extraction


class NewsScraper:
    """Scrapes financial news relevant to SGD from multiple sources."""

    def __init__(self, output_dir: str = "data/bronze/news"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "financial_news_2025.ndjson"
        self.seen_articles = self._load_seen_articles()

        # Expanded SGD-relevant keywords (broader FX context)
        self.sgd_keywords = {
            # Singapore-specific
            'singapore', 'sgd', 'singapore dollar', 'monetary authority of singapore',
            'mas', 'usd/sgd', 'usd_sgd', 'singapore economy', 'singapore gdp',
            'singapore inflation', 'singapore trade', 'singapore exports',
            'singapore central bank', 'singapore interest rates', 'neer',
            'nominal effective exchange rate', 'singapore banks', 'dbs', 'ocbc', 'uob',

            # ASEAN / Regional (affects SGD)
            'asean', 'southeast asia', 'asia pacific', 'asian markets',
            'regional trade', 'asian currencies', 'asia economy',

            # Global FX drivers (affects USD side of USD_SGD)
            'federal reserve', 'fed', 'fomc', 'dollar', 'usd', 'us dollar',
            'interest rate', 'rate hike', 'rate cut', 'monetary policy',
            'central bank', 'inflation', 'cpi', 'employment', 'gdp',

            # FX market general
            'currency', 'forex', 'fx', 'exchange rate', 'dollar strength',
            'safe haven', 'risk appetite', 'emerging markets'
        }

        # News sources configuration
        self.sources = [
            NewsSource(
                name="reuters_singapore",
                rss_url="https://www.reuters.com/markets/currencies/rss",
                base_url="https://www.reuters.com",
                selectors={
                    "content": "div[data-module='ArticleBody'] p",
                    "title": "h1",
                    "timestamp": "time"
                }
            ),
            NewsSource(
                name="bloomberg_asia",
                rss_url="https://feeds.bloomberg.com/markets/news.rss",
                base_url="https://www.bloomberg.com",
                selectors={
                    "content": ".body-content p",
                    "title": "h1",
                    "timestamp": "time"
                }
            ),
            NewsSource(
                name="channelnewsasia_business",
                rss_url="https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511",
                base_url="https://www.channelnewsasia.com",
                selectors={
                    "content": ".text-long p",
                    "title": "h1",
                    "timestamp": "time"
                }
            ),
            NewsSource(
                name="straits_times_business",
                rss_url="https://www.straitstimes.com/rss-feeds",
                base_url="https://www.straitstimes.com",
                selectors={
                    "content": ".story-content p",
                    "title": "h1",
                    "timestamp": "time"
                }
            )
        ]

    def _load_seen_articles(self) -> Set[str]:
        """Load previously seen article URLs to avoid duplicates."""
        seen = set()
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            article = json.loads(line)
                            seen.add(article.get('url', ''))
            except Exception as e:
                logger.warning(f"Error loading seen articles: {e}")
        return seen

    def _is_sgd_relevant(self, text: str) -> bool:
        """Check if article text is relevant to SGD."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.sgd_keywords)

    async def fetch_rss_feed(self, source: NewsSource) -> List[Dict]:
        """Fetch and parse RSS feed from a news source."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source.rss_url, timeout=30) as response:
                    if response.status == 200:
                        rss_content = await response.text()
                        feed = feedparser.parse(rss_content)

                        articles = []
                        for entry in feed.entries:
                            # Filter for recent articles (last 7 days for better coverage)
                            if hasattr(entry, 'published_parsed'):
                                pub_time = datetime(*entry.published_parsed[:6])
                                if datetime.now(timezone.utc).replace(tzinfo=None) - pub_time > timedelta(days=7):
                                    continue

                            articles.append({
                                'title': entry.title,
                                'url': entry.link,
                                'published': entry.get('published', ''),
                                'summary': entry.get('summary', ''),
                                'source': source.name
                            })

                        return articles

        except Exception as e:
            logger.error(f"Error fetching RSS from {source.name}: {e}")

        return []

    async def scrape_article_content(self, session: aiohttp.ClientSession,
                                   url: str, source: NewsSource) -> Optional[str]:
        """Scrape full article content from URL."""
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Extract content using source-specific selectors
                    content_elements = soup.select(source.selectors.get('content', 'p'))
                    content = ' '.join([elem.get_text().strip() for elem in content_elements])

                    return content if content else None

        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")

        return None

    async def process_articles(self, source: NewsSource) -> int:
        """Process all articles from a news source."""
        logger.info(f"Processing articles from {source.name}")

        # Fetch RSS feed
        rss_articles = await self.fetch_rss_feed(source)
        if not rss_articles:
            logger.warning(f"No articles found for {source.name}")
            return 0

        saved_count = 0
        async with aiohttp.ClientSession() as session:
            for article in rss_articles:
                url = article['url']

                # Skip if already processed
                if url in self.seen_articles:
                    continue

                # Check if title/summary is SGD relevant first (quick filter)
                quick_text = f"{article['title']} {article['summary']}"
                if not self._is_sgd_relevant(quick_text):
                    continue

                # Scrape full content
                content = await self.scrape_article_content(session, url, source)

                # Fallback: use RSS summary if scraping fails (paywalls/blocking)
                if not content:
                    if article['summary'] and len(article['summary']) > 50:
                        content = article['summary']
                        logger.info(f"Using RSS summary for: {article['title'][:60]}... (scraping blocked)")
                    else:
                        logger.warning(f"No content available for: {url}")
                        continue

                # Final relevance check on full content
                full_text = f"{article['title']} {article['summary']} {content}"
                if not self._is_sgd_relevant(full_text):
                    continue

                # Save to Bronze layer
                bronze_article = {
                    'url': url,
                    'title': article['title'],
                    'content': content,
                    'summary': article['summary'],
                    'published': article['published'],
                    'source': source.name,
                    'scraped_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    'sgd_relevant': True,
                    'word_count': len(content.split()),
                    'char_count': len(content)
                }

                # Append to NDJSON file
                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(bronze_article) + '\n')

                self.seen_articles.add(url)
                saved_count += 1

                logger.info(f"Saved: {article['title'][:80]}...")

                # Be respectful to servers
                await asyncio.sleep(1)

        return saved_count

    async def run_collection_cycle(self) -> Dict[str, int]:
        """Run one complete collection cycle across all sources."""
        logger.info("Starting news collection cycle")
        start_time = time.time()

        results = {}
        for source in self.sources:
            try:
                count = await self.process_articles(source)
                results[source.name] = count
                logger.info(f"{source.name}: {count} new SGD-relevant articles")

                # Wait between sources to be respectful
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error processing {source.name}: {e}")
                results[source.name] = 0

        elapsed = time.time() - start_time
        total_articles = sum(results.values())
        logger.info(f"Collection cycle completed: {total_articles} articles in {elapsed:.1f}s")

        return results

    async def start_live_collection(self, check_interval: int = 1800):
        """Start live news collection (default: every 30 minutes)."""
        logger.info("Starting live news collection")
        logger.info(f"Output file: {self.output_file}")
        logger.info(f"Check interval: {check_interval} seconds")

        while True:
            try:
                results = await self.run_collection_cycle()

                # Log collection summary
                total = sum(results.values())
                if total > 0:
                    logger.info(f"Collected {total} SGD-relevant articles this cycle")

                # Wait for next cycle
                logger.info(f"Waiting {check_interval}s for next collection cycle...")
                await asyncio.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("News collection stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in collection cycle: {e}")
                logger.info("Retrying in 5 minutes...")
                await asyncio.sleep(300)


async def main():
    """CLI entry point for news scraper."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape live SGD-relevant financial news")
    parser.add_argument("--output-dir", default="data/bronze/news", help="Output directory")
    parser.add_argument("--check-interval", type=int, default=1800,
                       help="Check interval in seconds (default: 30 minutes)")
    parser.add_argument("--test-run", action="store_true",
                       help="Run one collection cycle and exit")

    args = parser.parse_args()

    scraper = NewsScraper(output_dir=args.output_dir)

    if args.test_run:
        # Just run one cycle
        results = await scraper.run_collection_cycle()
        print(f"Test run completed: {results}")
    else:
        # Start live collection
        await scraper.start_live_collection(check_interval=args.check_interval)


if __name__ == "__main__":
    asyncio.run(main())