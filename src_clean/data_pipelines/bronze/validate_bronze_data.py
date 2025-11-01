import pandas as pd
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    args = parser.parse_args()
    
    print('Input data_file:', args.data_file)
    
    # Check market data
    market_file = Path(args.data_file)
    if not market_file.exists():
        print('ERROR: Market data not found at', market_file)
        sys.exit(1)

    # Load and validate
    df = pd.read_json(market_file, lines=True)
    if len(df) < 1000:
        print(f'WARNING: Only {len(df)} market rows found')
        sys.exit(1)

    print(f'✓ Market data validated: {len(df):,} rows')
    print(f'✓ Date range: {df.time.min()} to {df.time.max()}')
    print(f'✓ Columns: {list(df.columns)}')


if __name__ == "__main__":
    main()
