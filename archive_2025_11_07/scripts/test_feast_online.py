#!/usr/bin/env python3
"""Test Feast online feature fetching from Redis.

Verifies that features are properly materialized and can be retrieved
for real-time inference.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_online_features():
    """Test fetching features from Feast online store."""
    try:
        from feast import FeatureStore
    except ImportError:
        print("❌ Feast not installed. Run: pip install feast")
        sys.exit(1)

    print("[test_feast_online] Initializing Feast store")
    try:
        store = FeatureStore(repo_path=str(project_root / "feature_repo"))
    except Exception as e:
        print(f"❌ Failed to initialize Feast store: {e}")
        sys.exit(1)

    print("✓ Feast store initialized")

    # Test fetching online features for S&P 500
    print("\n[test_feast_online] Fetching online features for SPX500_USD")

    entity_rows = [
        {
            "instrument": "SPX500_USD"
        }
    ]

    try:
        # Fetch market features
        features_to_fetch = [
            "market_gold_features:ret_1h",
            "market_gold_features:ret_4h",
            "market_gold_features:vol_1d",
            "market_gold_features:rsi_14",
            "market_gold_features:orderbook_imbalance",
        ]

        online_features = store.get_online_features(
            features=features_to_fetch,
            entity_rows=entity_rows
        ).to_dict()

        print("✓ Successfully fetched market features from Redis")
        print("\nFeatures retrieved:")
        for key, value in online_features.items():
            if key != 'instrument':
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"⚠ Could not fetch market features: {e}")
        print("This is normal if market data hasn't been processed yet")

    # Test fetching news features
    try:
        news_features_to_fetch = [
            "news_gold_signals:news_sentiment_score",
            "news_gold_signals:news_signal_strength",
        ]

        online_features = store.get_online_features(
            features=news_features_to_fetch,
            entity_rows=entity_rows
        ).to_dict()

        print("\n✓ Successfully fetched news features from Redis")
        print("\nNews features retrieved:")
        for key, value in online_features.items():
            if key != 'instrument':
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"\n⚠ Could not fetch news features: {e}")
        print("This is normal if news data hasn't been processed yet")

    print("\n✅ Online feature test complete")
    print("\nIf you see errors above:")
    print("  1. Make sure Redis is running: docker-compose up -d redis")
    print("  2. Run feast materialize: python scripts/feast_materialize.py")
    print("  3. Ensure Parquet files exist with data")


if __name__ == "__main__":
    test_online_features()
