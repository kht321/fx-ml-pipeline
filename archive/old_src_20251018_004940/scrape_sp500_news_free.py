#!/usr/bin/env python3
"""
Free S&P 500 News Scraper (No API Keys Required)

Uses public RSS feeds and web scraping to collect S&P 500 news without needing API keys.
Sources:
1. Yahoo Finance RSS
2. MarketWatch RSS
3. CNBC RSS
4. Reuters Business RSS
5. Financial Times (where available)

Note: RSS feeds typically provide only recent articles (last 7-30 days).
For historical data, you'll need to run this daily or use paid APIs.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
import xml.etree.ElementTree as ET

try:
    import requests
    import feedparser
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install requests feedparser")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FreeNewsScraper:
    """Scrapes S&P 500 news from free public RSS feeds."""

    def __init__(self, output_dir: str = "data/news/bronze/raw_articles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Free RSS feeds
        self.feeds = {
            "yahoo_finance": "https://finance.yahoo.com/news/rss",
            "marketwatch": "https://www.marketwatch.com/rss/topstories",
            "cnbc_markets": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
            "reuters_business": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
            "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
        }

    def parse_rss_feed(self, feed_url: str, source_name: str) -> List[Dict]:
        """Parse an RSS feed and extract articles."""
        articles = []

        try:
            logger.info(f"Fetching {source_name} feed...")
            feed = feedparser.parse(feed_url)

            if not feed.entries:
                logger.warning(f"No entries found in {source_name} feed")
                return articles

            for entry in feed.entries:
                try:
                    # Extract published date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                    else:
                        pub_date = datetime.now(timezone.utc)

                    # Extract description/summary
                    description = ""
                    if hasattr(entry, 'summary'):
                        description = entry.summary
                    elif hasattr(entry, 'description'):
                        description = entry.description

                    article = {
                        "title": entry.get('title', 'No Title'),
                        "description": description,
                        "url": entry.get('link', ''),
                        "source": source_name,
                        "published_at": pub_date.isoformat(),
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        "category": entry.get('category', 'general'),
                    }

                    # Filter for S&P 500 related content
                    text = f"{article['title']} {article['description']}".lower()
                    if any(keyword in text for keyword in [
                        's&p', 'sp500', 's&p 500', 'stock market', 'stocks',
                        'wall street', 'nasdaq', 'dow', 'equity', 'market'
                    ]):
                        articles.append(article)

                except Exception as e:
                    logger.warning(f"Error parsing entry from {source_name}: {e}")
                    continue

            logger.info(f"✓ {source_name}: {len(articles)} relevant articles")

        except Exception as e:
            logger.error(f"Error fetching {source_name}: {e}")

        return articles

    def scrape_all_feeds(self) -> List[Dict]:
        """Scrape all configured RSS feeds."""
        all_articles = []

        logger.info("================================================================================")
        logger.info("Starting RSS feed scraping (no API keys required)...")
        logger.info("================================================================================")

        for source_name, feed_url in self.feeds.items():
            articles = self.parse_rss_feed(feed_url, source_name)
            all_articles.extend(articles)
            time.sleep(1)  # Be polite

        return all_articles

    def save_articles(self, articles: List[Dict]):
        """Save articles in NDJSON format."""
        if not articles:
            logger.warning("No articles to save")
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"sp500_news_free_{timestamp}.ndjson"

        with open(filename, 'w') as f:
            for article in articles:
                f.write(json.dumps(article) + '\n')

        logger.info(f"✓ Saved {len(articles)} articles to {filename}")

        # Create summary stats
        sources = {}
        for article in articles:
            source = article['source']
            sources[source] = sources.get(source, 0) + 1

        logger.info("\nArticles by source:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  • {source}: {count}")

    def run(self):
        """Main execution method."""
        try:
            articles = self.scrape_all_feeds()

            if articles:
                # Remove duplicates based on URL
                unique_articles = {a['url']: a for a in articles}.values()
                unique_articles = list(unique_articles)

                logger.info(f"\n✓ Total unique articles: {len(unique_articles)}")
                self.save_articles(unique_articles)

                logger.info("\n" + "="*80)
                logger.info("✅ Scraping complete!")
                logger.info("="*80)
                logger.info("\n💡 Note: RSS feeds only provide recent articles (7-30 days).")
                logger.info("💡 For historical data, run this script daily or use paid APIs.")
                logger.info("\n📊 Next steps:")
                logger.info("   1. Process news with FinGPT: python src/process_news_with_fingpt.py")
                logger.info("   2. Build news features: python src/build_news_features.py")

            else:
                logger.warning("No articles collected")

        except Exception as e:
            logger.error(f"Error during scraping: {e}", exc_info=True)


if __name__ == "__main__":
    scraper = FreeNewsScraper()
    scraper.run()
