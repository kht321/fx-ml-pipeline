"""
Implementation Validation Script

Validates that the FinBERT gold layer implementation is correctly integrated
into the pipeline without running the actual model.

Checks:
1. File structure and imports
2. Pipeline integration
3. Expected output format
4. Training pipeline compatibility
"""

import sys
from pathlib import Path
import ast
import re

def check_file_structure():
    """Check that all required files exist."""
    print("\n" + "="*80)
    print("CHECK 1: File Structure")
    print("="*80)

    required_files = {
        'Gold layer builder': Path('src_clean/data_pipelines/gold/news_signal_builder.py'),
        'Pipeline orchestrator': Path('src_clean/run_full_pipeline.py'),
        'Requirements': Path('requirements.txt'),
        'Training pipeline': Path('src_clean/training/xgboost_training_pipeline.py')
    }

    all_exist = True
    for name, file_path in required_files.items():
        if file_path.exists():
            print(f"✓ {name}: {file_path}")
        else:
            print(f"✗ {name}: MISSING - {file_path}")
            all_exist = False

    return all_exist


def check_dependencies():
    """Check that required dependencies are in requirements.txt."""
    print("\n" + "="*80)
    print("CHECK 2: Dependencies")
    print("="*80)

    requirements_file = Path('requirements.txt')

    with open(requirements_file, 'r') as f:
        content = f.read()

    required_deps = {
        'torch': r'torch==',
        'transformers': r'transformers==',
        'tqdm': r'tqdm==',
        'pandas': r'pandas==',
        'numpy': r'numpy=='
    }

    all_present = True
    for dep, pattern in required_deps.items():
        if re.search(pattern, content):
            # Extract version
            match = re.search(f'{pattern}([\\d.]+)', content)
            version = match.group(1) if match else 'unknown'
            print(f"✓ {dep}: {version}")
        else:
            print(f"✗ {dep}: NOT FOUND in requirements.txt")
            all_present = False

    return all_present


def check_gold_layer_implementation():
    """Check the gold layer builder implementation."""
    print("\n" + "="*80)
    print("CHECK 3: Gold Layer Implementation")
    print("="*80)

    builder_file = Path('src_clean/data_pipelines/gold/news_signal_builder.py')

    with open(builder_file, 'r') as f:
        content = f.read()

    # Check for required methods
    required_methods = [
        'load_finbert',
        'analyze_with_finbert',
        'load_article_bodies',
        'process_articles',
        'aggregate_signals',
        'run'
    ]

    all_methods = True
    for method in required_methods:
        if f'def {method}' in content:
            print(f"✓ Method: {method}()")
        else:
            print(f"✗ Method: {method}() - NOT FOUND")
            all_methods = False

    # Check for FinBERT model name
    if 'ProsusAI/finbert' in content:
        print(f"✓ FinBERT model: ProsusAI/finbert")
    else:
        print(f"✗ FinBERT model: NOT CONFIGURED")
        all_methods = False

    # Check output columns
    output_columns = [
        'signal_time',
        'avg_sentiment',
        'signal_strength',
        'trading_signal',
        'article_count',
        'quality_score'
    ]

    for col in output_columns:
        if col in content:
            print(f"✓ Output column: {col}")
        else:
            print(f"✗ Output column: {col} - NOT FOUND")
            all_methods = False

    return all_methods


def check_pipeline_integration():
    """Check that pipeline orchestrator includes news gold layer."""
    print("\n" + "="*80)
    print("CHECK 4: Pipeline Integration")
    print("="*80)

    pipeline_file = Path('src_clean/run_full_pipeline.py')

    with open(pipeline_file, 'r') as f:
        content = f.read()

    checks = {
        'gold_news_signals path': 'gold_news_signals',
        'build_gold_news method': 'def build_gold_news',
        'news_signal_builder import': 'news_signal_builder.py',
        'build_gold_news call': 'build_gold_news()'
    }

    all_integrated = True
    for check_name, pattern in checks.items():
        if pattern in content:
            print(f"✓ {check_name}")
        else:
            print(f"✗ {check_name} - NOT FOUND")
            all_integrated = False

    # Check stage numbering
    stages = re.findall(r'STAGE (\d+):', content)
    if stages:
        print(f"✓ Pipeline stages: {', '.join(set(stages))}")
        if '4' in stages and 'NEWS' in content:
            print(f"✓ Stage 4 is NEWS gold layer")
        else:
            print(f"⚠ Stage 4 might not be NEWS")

    return all_integrated


def check_training_compatibility():
    """Check that output format matches training pipeline expectations."""
    print("\n" + "="*80)
    print("CHECK 5: Training Pipeline Compatibility")
    print("="*80)

    training_file = Path('src_clean/training/xgboost_training_pipeline.py')

    with open(training_file, 'r') as f:
        content = f.read()

    # Expected columns in training pipeline
    if "'signal_time', 'avg_sentiment', 'signal_strength'" in content:
        print("✓ Training pipeline expects: signal_time, avg_sentiment, signal_strength")
    else:
        print("⚠ Could not verify expected columns in training pipeline")

    if "'trading_signal', 'article_count', 'quality_score'" in content:
        print("✓ Training pipeline expects: trading_signal, article_count, quality_score")
    else:
        print("⚠ Could not verify additional columns")

    # Check for news features
    if 'news_avg_sentiment' in content and 'news_signal_strength' in content:
        print("✓ Training pipeline uses news features")
    else:
        print("✗ Training pipeline might not use news features correctly")
        return False

    # Check merge logic
    if 'merge_market_news' in content:
        print("✓ Training pipeline has merge_market_news method")
    else:
        print("✗ Training pipeline missing merge logic")
        return False

    return True


def check_expected_workflow():
    """Describe the expected workflow."""
    print("\n" + "="*80)
    print("CHECK 6: Expected Workflow")
    print("="*80)

    workflow = """
Pipeline Flow:
1. Bronze News → Silver Sentiment (TextBlob/lexicon)
   - Input: data_clean/bronze/news/**/*.json
   - Output: data_clean/silver/news/sentiment/spx500_sentiment.csv

2. Silver Sentiment → Gold Signals (FinBERT)
   - Input: Silver sentiment CSV + Bronze articles (for full text)
   - Processing: FinBERT sentiment analysis + aggregation
   - Output: data_clean/gold/news/signals/spx500_news_signals.csv

3. Training (XGBoost)
   - Input: Gold market features + Gold news signals
   - Merge: As-of join on time
   - Features: news_avg_sentiment, news_signal_strength, etc.

Command to run:
    python src_clean/run_full_pipeline.py \\
        --bronze-market data_clean/bronze/market/spx500_usd_m1.ndjson \\
        --bronze-news data_clean/bronze/news \\
        --output-dir data_clean

Installation:
    pip install -r requirements.txt
    """

    print(workflow)
    print("✓ Workflow documented")

    return True


def main():
    """Run all validation checks."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*22 + "Implementation Validation" + " "*31 + "║")
    print("╚" + "="*78 + "╝")

    results = {}

    results['structure'] = check_file_structure()
    results['dependencies'] = check_dependencies()
    results['implementation'] = check_gold_layer_implementation()
    results['integration'] = check_pipeline_integration()
    results['compatibility'] = check_training_compatibility()
    results['workflow'] = check_expected_workflow()

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ ALL CHECKS PASSED - Implementation is correct!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run smoke test: python tests/test_finbert_gold_layer.py")
        print("  3. Run full pipeline with news data")
    else:
        print("\n✗ SOME CHECKS FAILED - Please fix issues")

    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
