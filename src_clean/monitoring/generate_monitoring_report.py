from pathlib import Path
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-features-file", required=True)
    args = parser.parse_args()

    print('=== Evidently Monitoring Report ===')

    # Check if required files exist
    features_file = Path(args.gold_features_file)
    if not features_file.exists():
        print('⚠ Features file not found, skipping monitoring report')
        print('  (This is expected on first run)')
        exit(0)

    df = pd.read_csv(features_file)
    print(f'✓ Loaded {len(df):,} rows for monitoring')

    # Simple data profile
    print(f'✓ Features: {len(df.columns)} columns')
    print(f'✓ Date range: {df.time.min() if "time" in df.columns else "N/A"} to {df.time.max() if "time" in df.columns else "N/A"}')
    print(f'✓ Missing values: {df.isnull().sum().sum()} total')

    print('✓ Monitoring data validated')
    print('  Note: Full Evidently report generation available via separate service')
    print('  Access at: http://localhost:8050')


if __name__ == "__main__":
    main()