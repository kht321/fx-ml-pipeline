#!/usr/bin/env python3
"""
Start 2025 Live Data Collection Pipeline

This script starts the complete data collection pipeline for 2025:
- Hourly USD_SGD candles from OANDA
- SGD-relevant financial news from multiple sources
- Integrated monitoring and health checks

Usage:
    python scripts/start_2025_collection.py
    python scripts/start_2025_collection.py --status
    python scripts/start_2025_collection.py --test-news
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_collection_pipeline import DataCollectionPipeline
from news_scraper import NewsScraper


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="2025 Live Data Collection Pipeline")
    parser.add_argument("--status", action="store_true",
                       help="Show current pipeline status")
    parser.add_argument("--test-news", action="store_true",
                       help="Test news collection for 5 minutes")
    parser.add_argument("--data-dir", default="data",
                       help="Data directory (default: data)")

    args = parser.parse_args()

    if args.status:
        # Show pipeline status
        pipeline = DataCollectionPipeline(data_dir=args.data_dir)
        status = pipeline.status()

        print("\n=== 2025 Live Data Collection Pipeline Status ===")
        print(f"📁 Data Directory: {status['data_directory']}")
        print(f"🔄 Pipeline Running: {status['pipeline_running']}")

        print("\n📈 Market Data (USD_SGD Hourly Candles):")
        market = status['market_data']
        if market.get('file_exists'):
            print(f"   ✅ File Size: {market['file_size_mb']:.1f} MB")
            print(f"   🕒 Last Candle: {market['last_candle'] or 'Unknown'}")
        else:
            print("   ❌ No market data file found")

        print(f"\n📰 News Data (SGD-relevant articles):")
        # Check news file manually since status doesn't include it yet
        news_file = Path(args.data_dir) / "bronze" / "news" / "financial_news_2025.ndjson"
        if news_file.exists():
            size_mb = news_file.stat().st_size / (1024 * 1024)
            print(f"   ✅ File Size: {size_mb:.1f} MB")

            # Count articles
            with open(news_file, 'r') as f:
                article_count = sum(1 for line in f if line.strip())
            print(f"   📄 Articles: {article_count}")
        else:
            print("   ❌ No news data file found")

        print(f"\n🛠️  To start collection: python {__file__}")
        print(f"📊 To test news only: python {__file__} --test-news")
        return

    elif args.test_news:
        # Test news collection for 5 minutes
        print("🧪 Testing news collection for 5 minutes...")
        scraper = NewsScraper(output_dir=f"{args.data_dir}/bronze/news")

        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 300:  # 5 minutes
            results = await scraper.run_collection_cycle()
            total = sum(results.values())
            print(f"✅ Collected {total} SGD-relevant articles: {results}")

            if asyncio.get_event_loop().time() - start_time < 300:
                print("⏳ Waiting 60 seconds before next cycle...")
                await asyncio.sleep(60)

        print("🏁 News collection test completed!")
        return

    else:
        # Start full pipeline
        print("🚀 Starting 2025 Live Data Collection Pipeline...")
        print("💡 Press Ctrl+C to stop gracefully")
        print()

        pipeline = DataCollectionPipeline(data_dir=args.data_dir)

        try:
            await pipeline.run()
        except KeyboardInterrupt:
            print("\n⏹️  Pipeline stopped by user")
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())