import pandas as pd
import sys
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-market-features-file", required=True)
    parser.add_argument("--gold-market-labels-file", required=True)
    parser.add_argument("--gold-news-signal-file", required=True)
    args = parser.parse_args()

    print('=== Gold Data Quality Validation ===')

    # Check market features
    features_file = Path(args.gold_market_features_file)
    if not features_file.exists():
        print(f'ERROR: Features file not found: {features_file}')
        sys.exit(1)

    df_features = pd.read_csv(features_file)
    print(f'✓ Features file loaded: {len(df_features):,} rows')

    # Check for missing values
    missing_pct = (df_features.isnull().sum() / len(df_features) * 100)
    critical_missing = missing_pct[missing_pct > 50]
    if len(critical_missing) > 0:
        print(f'ERROR: Critical missing values (>50%):')
        print(critical_missing)
        sys.exit(1)

    print(f'✓ Missing values check passed (max: {missing_pct.max():.2f}%)')

    # Check labels
    labels_file = Path(args.gold_market_labels_file)
    if not labels_file.exists():
        print(f'ERROR: Labels file not found: {labels_file}')
        sys.exit(1)

    df_labels = pd.read_csv(labels_file)
    print(f'✓ Labels file loaded: {len(df_labels):,} rows')

    # Check news signals
    news_file = Path(args.gold_news_signal_file)
    if news_file.exists():
        df_news = pd.read_csv(news_file)
        print(f'✓ News signals loaded: {len(df_news):,} rows')
    else:
        print('⚠ News signals file not found (optional)')

    print('✓ Gold data quality validation PASSED')


if __name__ == "__main__":
    main()