#!/usr/bin/env python3
"""
S&P 500 Complete Pipeline Runner

This script runs the entire medallion architecture pipeline for S&P 500 data:
1. Bronze Layer: Raw OHLCV candle data
2. Silver Layer: Technical features, microstructure, volatility
3. Gold Layer: Training-ready features with labels

Then exports all features to CSV for analysis.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import pandas as pd


def log(message: str, level: str = "INFO"):
    """Structured logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def run_command(cmd: List[str], description: str, check=True) -> bool:
    """Run a shell command with logging."""
    log(f"Running: {description}")
    log(f"Command: {' '.join(cmd)}", "DEBUG")

    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)

        if result.returncode == 0:
            log(f"✓ {description} completed", "SUCCESS")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            log(f"✗ {description} failed", "ERROR")
            if result.stderr.strip():
                print(result.stderr, file=sys.stderr)
            return False

    except subprocess.CalledProcessError as e:
        log(f"✗ {description} failed with error", "ERROR")
        print(e.stderr, file=sys.stderr)
        return False
    except Exception as e:
        log(f"✗ Unexpected error: {e}", "ERROR")
        return False


def setup_directories():
    """Create necessary directory structure."""
    log("Setting up directory structure...")

    dirs = [
        "data/sp500/bronze/prices",
        "data/sp500/silver/technical_features",
        "data/sp500/silver/microstructure",
        "data/sp500/silver/volatility",
        "data/sp500/gold/training",
        "data/sp500/gold/features",
        "outputs/feature_csvs",
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    log("✓ Directories created")


def check_bronze_data() -> Path:
    """Check for bronze S&P 500 data and return path."""
    log("Checking for S&P 500 bronze data...")

    # Check for the 5-year dataset first, then 2-year, then 1-year
    candidates = [
        "data/bronze/prices/spx500_usd_m1_5years.ndjson",
        "data/bronze/prices/spx500_usd_m1_historical.ndjson",
        "data/bronze/prices/spx500_usd_m1_2years.ndjson",
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.stat().st_size > 0:
            log(f"✓ Found bronze data: {path}")
            log(f"  Size: {path.stat().st_size / (1024**2):.1f} MB")

            # Count candles
            with open(path) as f:
                candle_count = sum(1 for line in f if line.strip())
            log(f"  Candles: {candle_count:,}")

            return path

    log("✗ No S&P 500 bronze data found!", "ERROR")
    log("  Please run: python src/download_sp500_historical.py --years 5", "ERROR")
    sys.exit(1)


def run_bronze_to_silver(bronze_path: Path, sample_size: int = None) -> Dict[str, Path]:
    """Transform Bronze OHLCV data to Silver technical features."""
    log("=" * 80)
    log("STEP 1: Bronze → Silver (Feature Engineering)")
    log("=" * 80)

    # Prepare input (optionally sample for faster testing)
    input_path = bronze_path

    if sample_size:
        log(f"Sampling {sample_size:,} candles for testing...")
        sample_path = Path("data/sp500/bronze/prices/sp500_sample.ndjson")
        with open(bronze_path) as infile, open(sample_path, 'w') as outfile:
            for i, line in enumerate(infile):
                if i >= sample_size:
                    break
                outfile.write(line)
        input_path = sample_path
        log(f"✓ Sample created: {sample_path}")

    output_paths = {
        'technical': Path("data/sp500/silver/technical_features/sp500_technical.csv"),
        'microstructure': Path("data/sp500/silver/microstructure/sp500_microstructure.csv"),
        'volatility': Path("data/sp500/silver/volatility/sp500_volatility.csv"),
    }

    cmd = [
        sys.executable, "src/build_market_features_from_candles.py",
        "--input", str(input_path),
        "--output-technical", str(output_paths['technical']),
        "--output-microstructure", str(output_paths['microstructure']),
        "--output-volatility", str(output_paths['volatility']),
        "--min-rows", "50",
    ]

    success = run_command(cmd, "Bronze → Silver transformation")

    if success:
        log("\nSilver Layer Created:")
        for name, path in output_paths.items():
            if path.exists():
                df = pd.read_csv(path)
                log(f"  ✓ {name:15s}: {len(df):,} rows, {len(df.columns)} columns")
            else:
                log(f"  ✗ {name:15s}: NOT CREATED", "WARNING")

    return output_paths if success else {}


def run_silver_to_gold(silver_paths: Dict[str, Path]) -> Path:
    """Consolidate Silver features into Gold training data."""
    log("=" * 80)
    log("STEP 2: Silver → Gold (Feature Consolidation)")
    log("=" * 80)

    output_path = Path("data/sp500/gold/training/sp500_features.csv")

    cmd = [
        sys.executable, "src/build_sp500_gold.py",
        "--technical-features", str(silver_paths['technical']),
        "--microstructure-features", str(silver_paths['microstructure']),
        "--volatility-features", str(silver_paths['volatility']),
        "--output", str(output_path),
        "--feature-selection", "all",
    ]

    success = run_command(cmd, "Silver → Gold consolidation")

    if success and output_path.exists():
        df = pd.read_csv(output_path)
        log(f"\n✓ Gold Layer Created: {len(df):,} rows, {len(df.columns)} columns")
        log(f"  Output: {output_path}")
        return output_path

    return None


def add_labels(gold_path: Path) -> Path:
    """Add training labels to Gold features."""
    log("=" * 80)
    log("STEP 3: Adding Training Labels")
    log("=" * 80)

    output_path = Path("data/sp500/gold/training/sp500_features_with_labels.csv")

    cmd = [
        sys.executable, "src/build_labels.py",
        "--input", str(gold_path),
        "--output", str(output_path),
        "--horizon", "5",  # 5-minute ahead prediction
        "--threshold", "0.001",  # 0.1% move threshold
    ]

    success = run_command(cmd, "Label generation")

    if success and output_path.exists():
        df = pd.read_csv(output_path)
        log(f"\n✓ Labels Added: {len(df):,} rows, {len(df.columns)} columns")

        # Show label distribution
        if 'label' in df.columns:
            label_dist = df['label'].value_counts()
            log("\nLabel Distribution:")
            for label, count in label_dist.items():
                pct = count / len(df) * 100
                log(f"  {label:10s}: {count:,} ({pct:.1f}%)")

        return output_path

    return None


def export_features_to_csv(gold_path: Path, output_dir: Path):
    """Export features to CSV with metadata."""
    log("=" * 80)
    log("STEP 4: Exporting Features to CSV")
    log("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the gold data
    df = pd.read_csv(gold_path)

    # 1. Export full feature set
    full_output = output_dir / "sp500_all_features.csv"
    df.to_csv(full_output, index=False)
    log(f"✓ Full feature set: {full_output}")
    log(f"  {len(df):,} rows × {len(df.columns)} columns")

    # 2. Create feature metadata
    feature_metadata = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100

        if dtype in ['float64', 'int64']:
            stats = {
                'feature_name': col,
                'data_type': dtype,
                'missing_count': missing,
                'missing_pct': f"{missing_pct:.2f}%",
                'min': df[col].min() if missing < len(df) else None,
                'max': df[col].max() if missing < len(df) else None,
                'mean': df[col].mean() if missing < len(df) else None,
                'std': df[col].std() if missing < len(df) else None,
                'unique_values': df[col].nunique(),
            }
        else:
            stats = {
                'feature_name': col,
                'data_type': dtype,
                'missing_count': missing,
                'missing_pct': f"{missing_pct:.2f}%",
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
                'unique_values': df[col].nunique(),
            }

        # Categorize feature
        if 'time' in col.lower():
            category = 'temporal'
        elif any(x in col.lower() for x in ['ret', 'return', 'pct']):
            category = 'returns'
        elif any(x in col.lower() for x in ['vol', 'volatility', 'std']):
            category = 'volatility'
        elif any(x in col.lower() for x in ['sma', 'ema', 'ma']):
            category = 'moving_average'
        elif any(x in col.lower() for x in ['rsi', 'macd', 'bb']):
            category = 'technical_indicator'
        elif 'label' in col.lower():
            category = 'target'
        else:
            category = 'other'

        stats['category'] = category
        feature_metadata.append(stats)

    metadata_df = pd.DataFrame(feature_metadata)
    metadata_output = output_dir / "sp500_feature_metadata.csv"
    metadata_df.to_csv(metadata_output, index=False)
    log(f"✓ Feature metadata: {metadata_output}")

    # 3. Export by category
    categories = metadata_df['category'].unique()
    for category in categories:
        cat_features = metadata_df[metadata_df['category'] == category]['feature_name'].tolist()
        cat_df = df[cat_features]
        cat_output = output_dir / f"sp500_features_{category}.csv"
        cat_df.to_csv(cat_output, index=False)
        log(f"✓ {category} features: {cat_output} ({len(cat_features)} features)")

    # 4. Create summary report
    summary = {
        'pipeline_run_date': datetime.now().isoformat(),
        'total_samples': len(df),
        'total_features': len(df.columns),
        'feature_categories': {
            cat: len(metadata_df[metadata_df['category'] == cat])
            for cat in categories
        },
        'date_range': {
            'start': df['time'].min() if 'time' in df.columns else 'N/A',
            'end': df['time'].max() if 'time' in df.columns else 'N/A',
        },
        'missing_data_summary': {
            'features_with_missing': int((metadata_df['missing_count'] > 0).sum()),
            'total_missing_values': int(metadata_df['missing_count'].sum()),
        }
    }

    summary_output = output_dir / "sp500_feature_summary.json"
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"✓ Summary report: {summary_output}")

    # Print summary
    log("\n" + "=" * 80)
    log("FEATURE EXPORT SUMMARY")
    log("=" * 80)
    log(f"Total Samples:  {summary['total_samples']:,}")
    log(f"Total Features: {summary['total_features']:,}")
    log(f"\nFeatures by Category:")
    for cat, count in summary['feature_categories'].items():
        log(f"  {cat:20s}: {count:3d} features")
    log(f"\nDate Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    log(f"\nOutput Directory: {output_dir}")
    log("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sample',
        type=int,
        help="Use only first N candles for testing (default: use all data)"
    )
    parser.add_argument(
        '--skip-bronze-silver',
        action='store_true',
        help="Skip Bronze→Silver step (use existing Silver data)"
    )
    parser.add_argument(
        '--skip-silver-gold',
        action='store_true',
        help="Skip Silver→Gold step (use existing Gold data)"
    )
    parser.add_argument(
        '--skip-labels',
        action='store_true',
        help="Skip label generation"
    )

    args = parser.parse_args()

    log("=" * 80)
    log("S&P 500 COMPLETE PIPELINE")
    log("=" * 80)
    log(f"Start Time: {datetime.now()}")
    log("")

    # Setup
    setup_directories()

    # Check bronze data
    bronze_path = check_bronze_data()

    # Step 1: Bronze → Silver
    if not args.skip_bronze_silver:
        silver_paths = run_bronze_to_silver(bronze_path, args.sample)
        if not silver_paths:
            log("Pipeline failed at Bronze→Silver step", "ERROR")
            sys.exit(1)
    else:
        log("Skipping Bronze→Silver (using existing Silver data)")
        silver_paths = {
            'technical': Path("data/sp500/silver/technical_features/sp500_technical.csv"),
            'microstructure': Path("data/sp500/silver/microstructure/sp500_microstructure.csv"),
            'volatility': Path("data/sp500/silver/volatility/sp500_volatility.csv"),
        }

    # Step 2: Silver → Gold
    if not args.skip_silver_gold:
        gold_path = run_silver_to_gold(silver_paths)
        if not gold_path:
            log("Pipeline failed at Silver→Gold step", "ERROR")
            sys.exit(1)
    else:
        log("Skipping Silver→Gold (using existing Gold data)")
        gold_path = Path("data/sp500/gold/training/sp500_features.csv")

    # Step 3: Add labels
    if not args.skip_labels:
        gold_with_labels = add_labels(gold_path)
        if not gold_with_labels:
            log("Pipeline failed at label generation", "ERROR")
            sys.exit(1)
        final_path = gold_with_labels
    else:
        log("Skipping label generation")
        final_path = gold_path

    # Step 4: Export features
    export_features_to_csv(final_path, Path("outputs/feature_csvs"))

    log("\n" + "=" * 80)
    log("✓ PIPELINE COMPLETE!")
    log("=" * 80)
    log(f"End Time: {datetime.now()}")
    log("\nNext steps:")
    log("  1. Review features: outputs/feature_csvs/sp500_feature_metadata.csv")
    log("  2. Train models using: data/sp500/gold/training/sp500_features_with_labels.csv")
    log("  3. Check summary: outputs/feature_csvs/sp500_feature_summary.json")


if __name__ == "__main__":
    main()
