"""
Analyze data coverage and timestamp alignment between market and news data.

Repository: fx-ml-pipeline/analyze_data_coverage.py
Purpose: Assess data quality, timezone consistency, and coverage gaps
"""

import json
import glob
from datetime import datetime
from dateutil import parser
import pytz

def analyze_market_data():
    """Analyze market data coverage and timestamps."""
    print("=" * 80)
    print("MARKET DATA ANALYSIS")
    print("=" * 80)

    market_file = "data/bronze/prices/spx500_usd_m1_5years.ndjson"

    with open(market_file, 'r') as f:
        first_line = json.loads(f.readline())

        # Count total lines
        f.seek(0)
        line_count = sum(1 for _ in f)

        # Get last line
        f.seek(0)
        for i, line in enumerate(f):
            if i == line_count - 1:
                last_line = json.loads(line)
                break

    # Parse timestamps (handle nanoseconds)
    first_time_str = first_line['time'].replace('Z', '').split('.')[0]
    last_time_str = last_line['time'].replace('Z', '').split('.')[0]

    first_dt = datetime.fromisoformat(first_time_str).replace(tzinfo=pytz.UTC)
    last_dt = datetime.fromisoformat(last_time_str).replace(tzinfo=pytz.UTC)

    days_span = (last_dt - first_dt).days
    years_span = days_span / 365.25

    print(f"File: {market_file}")
    print(f"Total candles: {line_count:,}")
    print(f"First candle: {first_dt} UTC")
    print(f"Last candle:  {last_dt} UTC")
    print(f"Date range: {days_span} days ({years_span:.2f} years)")
    print(f"Sample: {first_line}")

    # Calculate expected vs actual candles
    trading_days_per_year = 252
    hours_per_trading_day = 6.5  # US market hours
    minutes_per_year = trading_days_per_year * hours_per_trading_day * 60
    expected_candles = int(years_span * minutes_per_year)

    coverage_pct = (line_count / expected_candles) * 100 if expected_candles > 0 else 0

    print(f"Expected candles (approx): {expected_candles:,}")
    print(f"Actual candles: {line_count:,}")
    print(f"Coverage: {coverage_pct:.1f}%")

    return first_dt, last_dt


def analyze_news_data():
    """Analyze news data coverage and timestamps."""
    print("\n" + "=" * 80)
    print("NEWS DATA ANALYSIS")
    print("=" * 80)

    news_files = sorted(glob.glob('data/bronze/news/articles/*.json'))

    if not news_files:
        print("No news articles found!")
        return None, None

    print(f"Total news articles: {len(news_files)}")

    timestamps = []
    sources = set()

    for nf in news_files:
        with open(nf, 'r') as f:
            article = json.load(f)
            pub_time_str = article.get('published_at')

            # Parse various timestamp formats
            try:
                pub_dt = parser.parse(pub_time_str)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=pytz.UTC)
                timestamps.append(pub_dt)
                sources.add(article.get('source', 'unknown'))
            except Exception as e:
                print(f"Could not parse timestamp: {pub_time_str} - {e}")

    if timestamps:
        first_news = min(timestamps)
        last_news = max(timestamps)
        news_span_days = (last_news - first_news).days

        print(f"First article: {first_news} UTC")
        print(f"Last article:  {last_news} UTC")
        print(f"News span: {news_span_days} days ({news_span_days / 365.25:.2f} years)")
        print(f"Sources: {', '.join(sorted(sources))}")

        # Sample timestamps
        print("\nSample article timestamps:")
        for nf in news_files[:3]:
            with open(nf, 'r') as f:
                article = json.load(f)
                print(f"  {article.get('published_at')} - {article.get('headline')[:60]}...")

        return first_news, last_news

    return None, None


def check_timezone_alignment(market_first, market_last, news_first, news_last):
    """Check if market and news data are in the same timezone and aligned."""
    print("\n" + "=" * 80)
    print("TIMEZONE AND ALIGNMENT CHECK")
    print("=" * 80)

    if news_first is None or news_last is None:
        print("Cannot check alignment - insufficient news data")
        return

    print(f"Market timezone: {market_first.tzinfo}")
    print(f"News timezone:   {news_first.tzinfo}")

    # Check overlap
    overlap_start = max(market_first, news_first)
    overlap_end = min(market_last, news_last)

    if overlap_start < overlap_end:
        overlap_days = (overlap_end - overlap_start).days
        print(f"\n✓ Data overlap exists!")
        print(f"  Overlap period: {overlap_start} to {overlap_end}")
        print(f"  Overlap duration: {overlap_days} days ({overlap_days / 365.25:.2f} years)")
    else:
        print(f"\n✗ NO DATA OVERLAP!")
        print(f"  Market data: {market_first} to {market_last}")
        print(f"  News data:   {news_first} to {news_last}")
        print(f"  Gap: {(news_first - market_last).days} days" if news_first > market_last else f"  Gap: {(market_first - news_last).days} days")

    # Calculate coverage gap
    market_days = (market_last - market_first).days
    news_days = (news_last - news_first).days

    print(f"\nCoverage comparison:")
    print(f"  Market data covers: {market_days / 365.25:.2f} years")
    print(f"  News data covers:   {news_days / 365.25:.2f} years")

    if news_days < market_days:
        missing_years = (market_days - news_days) / 365.25
        print(f"\n⚠ News data gap: {missing_years:.2f} years of data missing")
        print(f"  Recommendation: Scrape {missing_years:.1f} more years of historical news")


def assess_news_sufficiency():
    """Assess if we have enough news articles for training."""
    print("\n" + "=" * 80)
    print("NEWS DATA SUFFICIENCY ASSESSMENT")
    print("=" * 80)

    news_files = glob.glob('data/bronze/news/articles/*.json')
    article_count = len(news_files)

    # Benchmarks for good ML training
    min_articles_per_day = 5
    optimal_articles_per_day = 20

    # For 5 years
    target_days = 5 * 365
    min_needed = target_days * min_articles_per_day
    optimal_needed = target_days * optimal_articles_per_day

    print(f"Current articles: {article_count:,}")
    print(f"Minimum needed (5 articles/day for 5 years): {min_needed:,}")
    print(f"Optimal (20 articles/day for 5 years): {optimal_needed:,}")

    if article_count < min_needed:
        shortfall = min_needed - article_count
        print(f"\n✗ INSUFFICIENT DATA")
        print(f"  Need at least {shortfall:,} more articles")
        print(f"  Current coverage: {(article_count / min_needed) * 100:.1f}% of minimum")
        print(f"\n  RECOMMENDATION: Scrape historical news data urgently!")
    elif article_count < optimal_needed:
        print(f"\n△ MINIMAL DATA (can train but not optimal)")
        print(f"  Current coverage: {(article_count / optimal_needed) * 100:.1f}% of optimal")
        print(f"  Recommendation: Add more news sources for better model performance")
    else:
        print(f"\n✓ SUFFICIENT DATA")
        print(f"  Coverage: {(article_count / optimal_needed) * 100:.1f}% of optimal")


def main():
    """Main analysis function."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DATA COVERAGE & ALIGNMENT ANALYSIS" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Analyze market data
    market_first, market_last = analyze_market_data()

    # Analyze news data
    news_first, news_last = analyze_news_data()

    # Check alignment
    check_timezone_alignment(market_first, market_last, news_first, news_last)

    # Assess sufficiency
    assess_news_sufficiency()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
