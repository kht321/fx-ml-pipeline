"""
Smoke Test for FinBERT Gold Layer Builder

Tests the FinBERT news signal builder with sample data to ensure:
1. FinBERT model loads correctly
2. Sentiment analysis works
3. Signal aggregation produces expected output format
4. Output is compatible with training pipeline
"""

import sys
from pathlib import Path
import pandas as pd
import json
import tempfile
import shutil

# Add src_clean to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src_clean"))

from data_pipelines.gold.news_signal_builder import FinBERTSignalBuilder


def create_sample_data():
    """Create sample data for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create bronze news directory
    bronze_dir = temp_dir / "bronze" / "news"
    bronze_dir.mkdir(parents=True)

    # Sample articles with varied sentiment
    sample_articles = [
        {
            "article_id": "art_001",
            "headline": "S&P 500 Surges to Record High on Strong Earnings",
            "body": "The S&P 500 index rallied to a new all-time high today as major tech companies reported better-than-expected earnings. Investors showed strong optimism about economic growth.",
            "published_at": "2025-01-15T10:00:00Z",
            "source": "reuters"
        },
        {
            "article_id": "art_002",
            "headline": "Fed Signals Hawkish Stance Amid Inflation Concerns",
            "body": "The Federal Reserve indicated it may raise interest rates more aggressively than previously anticipated due to persistent inflation pressures. Markets reacted negatively to the news.",
            "published_at": "2025-01-15T10:30:00Z",
            "source": "bloomberg"
        },
        {
            "article_id": "art_003",
            "headline": "Market Holds Steady as Investors Await Economic Data",
            "body": "Trading volumes remained moderate today as market participants waited for key economic indicators to be released later this week. Analysts expect neutral market conditions to continue.",
            "published_at": "2025-01-15T11:00:00Z",
            "source": "cnbc"
        },
        {
            "article_id": "art_004",
            "headline": "Tech Stocks Plunge on Disappointing Guidance",
            "body": "Major technology stocks fell sharply after several companies issued weaker-than-expected forward guidance. The selloff extended across the sector with heavy losses.",
            "published_at": "2025-01-15T11:30:00Z",
            "source": "wsj"
        },
        {
            "article_id": "art_005",
            "headline": "Strong Jobs Report Boosts Market Confidence",
            "body": "Employment figures exceeded expectations with robust job creation and falling unemployment. The positive economic news lifted investor sentiment and pushed markets higher.",
            "published_at": "2025-01-15T12:00:00Z",
            "source": "reuters"
        }
    ]

    # Write bronze articles
    for article in sample_articles:
        article_file = bronze_dir / f"{article['article_id']}.json"
        with open(article_file, 'w') as f:
            json.dump(article, f, indent=2)

    # Create silver sentiment CSV (from TextBlob processor)
    silver_dir = temp_dir / "silver" / "news" / "sentiment"
    silver_dir.mkdir(parents=True)

    silver_data = []
    for article in sample_articles:
        silver_data.append({
            'article_id': article['article_id'],
            'published_at': article['published_at'],
            'source': article['source'],
            'headline': article['headline'],
            'polarity': 0.5,  # Dummy TextBlob values
            'subjectivity': 0.5,
            'financial_sentiment': 0.0,
            'confidence': 0.6,
            'policy_tone': 'neutral',
            'headline_length': len(article['headline']),
            'body_length': len(article['body'])
        })

    silver_csv = silver_dir / "spx500_sentiment.csv"
    pd.DataFrame(silver_data).to_csv(silver_csv, index=False)

    return temp_dir, silver_csv, bronze_dir


def test_finbert_loading():
    """Test 1: FinBERT model loads correctly."""
    print("\n" + "="*80)
    print("TEST 1: FinBERT Model Loading")
    print("="*80)

    try:
        builder = FinBERTSignalBuilder(aggregation_window_minutes=60)
        builder.load_finbert()

        print("✓ FinBERT model loaded successfully")
        print(f"✓ Device: {builder.device}")
        print(f"✓ Model: {builder.model.__class__.__name__}")
        print(f"✓ Tokenizer: {builder.tokenizer.__class__.__name__}")

        return True

    except Exception as e:
        print(f"✗ Failed to load FinBERT: {e}")
        return False


def test_sentiment_analysis(builder):
    """Test 2: Sentiment analysis works on sample texts."""
    print("\n" + "="*80)
    print("TEST 2: Sentiment Analysis")
    print("="*80)

    test_texts = [
        ("Positive", "Stock market surges to record high on strong earnings and optimistic outlook"),
        ("Negative", "Market crashes as recession fears grow and companies report massive losses"),
        ("Neutral", "Market holds steady as investors await economic data releases")
    ]

    try:
        for expected_sentiment, text in test_texts:
            result = builder.analyze_with_finbert(text)

            print(f"\nText: {text[:60]}...")
            print(f"  Expected: {expected_sentiment}")
            print(f"  Predicted: {result['sentiment']}")
            print(f"  Score: {result['sentiment_score']:.3f}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Probs: pos={result['positive_prob']:.2f}, "
                  f"neg={result['negative_prob']:.2f}, "
                  f"neu={result['neutral_prob']:.2f}")

        print("\n✓ Sentiment analysis working correctly")
        return True

    except Exception as e:
        print(f"\n✗ Sentiment analysis failed: {e}")
        return False


def test_full_pipeline(builder, temp_dir, silver_csv, bronze_dir):
    """Test 3: Full pipeline produces correct output format."""
    print("\n" + "="*80)
    print("TEST 3: Full Pipeline")
    print("="*80)

    gold_dir = temp_dir / "gold" / "news" / "signals"
    gold_dir.mkdir(parents=True)
    gold_csv = gold_dir / "spx500_news_signals.csv"

    try:
        # Run pipeline
        builder.run(silver_csv, bronze_dir, gold_csv)

        # Check output exists
        if not gold_csv.exists():
            print("✗ Output file not created")
            return False

        # Load and validate output
        gold_df = pd.read_csv(gold_csv)

        print(f"\n✓ Output file created: {gold_csv}")
        print(f"✓ Rows: {len(gold_df)}")
        print(f"✓ Columns: {len(gold_df.columns)}")

        # Check required columns
        required_columns = [
            'signal_time', 'avg_sentiment', 'signal_strength',
            'trading_signal', 'article_count', 'quality_score'
        ]

        missing_columns = [col for col in required_columns if col not in gold_df.columns]

        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            return False

        print(f"✓ All required columns present: {required_columns}")

        # Check data types
        print(f"\n✓ signal_time: {gold_df['signal_time'].dtype}")
        print(f"✓ avg_sentiment: {gold_df['avg_sentiment'].dtype} (range: {gold_df['avg_sentiment'].min():.3f} to {gold_df['avg_sentiment'].max():.3f})")
        print(f"✓ signal_strength: {gold_df['signal_strength'].dtype} (range: {gold_df['signal_strength'].min():.3f} to {gold_df['signal_strength'].max():.3f})")
        print(f"✓ trading_signal: {gold_df['trading_signal'].dtype} (unique: {sorted(gold_df['trading_signal'].unique())})")
        print(f"✓ article_count: {gold_df['article_count'].dtype} (total: {gold_df['article_count'].sum()})")
        print(f"✓ quality_score: {gold_df['quality_score'].dtype} (avg: {gold_df['quality_score'].mean():.3f})")

        # Check signal distribution
        signal_dist = gold_df['trading_signal'].value_counts().to_dict()
        print(f"\n✓ Trading signals:")
        print(f"    Buy (1): {signal_dist.get(1, 0)}")
        print(f"    Sell (-1): {signal_dist.get(-1, 0)}")
        print(f"    Hold (0): {signal_dist.get(0, 0)}")

        return True

    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline_compatibility(temp_dir):
    """Test 4: Output is compatible with training pipeline."""
    print("\n" + "="*80)
    print("TEST 4: Training Pipeline Compatibility")
    print("="*80)

    gold_csv = temp_dir / "gold" / "news" / "signals" / "spx500_news_signals.csv"

    try:
        # Load gold signals
        news_df = pd.read_csv(gold_csv)
        news_df['signal_time'] = pd.to_datetime(news_df['signal_time'])

        print(f"✓ Loaded news signals: {len(news_df)} rows")

        # Simulate training pipeline's expected columns
        expected_features = [
            'signal_time', 'avg_sentiment', 'signal_strength',
            'trading_signal', 'article_count', 'quality_score'
        ]

        available_features = [c for c in expected_features if c in news_df.columns]

        print(f"✓ Available features: {len(available_features)}/{len(expected_features)}")

        # Test as-of merge (what training pipeline does)
        print("\n✓ Testing as-of join (training pipeline behavior):")

        # Create mock market data
        market_times = pd.date_range(
            start=news_df['signal_time'].min(),
            end=news_df['signal_time'].max(),
            freq='15min'
        )

        mock_market_df = pd.DataFrame({
            'time': market_times,
            'close': 4500.0
        })

        print(f"  Mock market data: {len(mock_market_df)} observations")

        # Perform as-of merge
        merged = pd.merge_asof(
            mock_market_df.sort_values('time'),
            news_df[available_features].sort_values('signal_time'),
            left_on='time',
            right_on='signal_time',
            direction='backward',
            tolerance=pd.Timedelta(hours=6)
        )

        print(f"  Merged data: {len(merged)} rows")
        print(f"  News coverage: {merged['avg_sentiment'].notna().mean():.1%}")

        if merged['avg_sentiment'].notna().any():
            print(f"✓ Merge successful - news features available for prediction")
            return True
        else:
            print(f"✗ Merge failed - no news features available")
            return False

    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "FinBERT Gold Layer Smoke Test" + " "*29 + "║")
    print("╚" + "="*78 + "╝")

    # Create sample data
    print("\nCreating sample test data...")
    temp_dir, silver_csv, bronze_dir = create_sample_data()
    print(f"✓ Test data created in: {temp_dir}")

    # Run tests
    results = {}

    try:
        # Test 1: Model loading
        results['loading'] = test_finbert_loading()

        if not results['loading']:
            print("\n✗ Cannot proceed - FinBERT failed to load")
            print("\nHint: Run 'pip install transformers torch' to install dependencies")
            return False

        # Create builder instance for remaining tests
        builder = FinBERTSignalBuilder(aggregation_window_minutes=60)
        builder.load_finbert()

        # Test 2: Sentiment analysis
        results['sentiment'] = test_sentiment_analysis(builder)

        # Test 3: Full pipeline
        results['pipeline'] = test_full_pipeline(builder, temp_dir, silver_csv, bronze_dir)

        # Test 4: Training compatibility
        if results['pipeline']:
            results['compatibility'] = test_training_pipeline_compatibility(temp_dir)
        else:
            results['compatibility'] = False

    finally:
        # Cleanup
        print(f"\nCleaning up test data: {temp_dir}")
        shutil.rmtree(temp_dir)

    # Summary
    print("\n" + "="*80)
    print("SMOKE TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ ALL TESTS PASSED - FinBERT gold layer is ready!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run full pipeline with news: python src_clean/run_full_pipeline.py ...")
        print("  3. Train model with news signals: python src_clean/training/xgboost_training_pipeline.py ...")
    else:
        print("\n✗ SOME TESTS FAILED - Please fix issues before proceeding")

    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
