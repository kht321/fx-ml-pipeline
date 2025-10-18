"""
Data Migration Script - Restructure to Clean Medallion Architecture

Repository Location: fx-ml-pipeline/migrate_to_clean_structure.py

Purpose:
    Migrates existing data from the old structure to the new clean medallion architecture:

    OLD Structure:
        data/bronze/, data/sp500/, data/news/, data/market/

    NEW Structure:
        data_clean/
        ├── bronze/
        │   ├── market/  (raw OHLCV candles)
        │   └── news/    (raw articles)
        ├── silver/
        │   ├── market/  (technical, microstructure, volatility features)
        │   └── news/    (sentiment, entities, topics)
        └── gold/
            ├── market/  (training-ready features)
            └── news/    (trading signals)

Usage:
    python migrate_to_clean_structure.py --execute
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataMigrator:
    """Handles migration of data to new clean structure."""

    def __init__(self, dry_run: bool = True):
        """
        Initialize migrator.

        Parameters
        ----------
        dry_run : bool
            If True, only show what would be done without actually moving files
        """
        self.dry_run = dry_run
        self.base_dir = Path.cwd()

        # Old paths
        self.old_bronze = self.base_dir / "data" / "bronze"
        self.old_sp500 = self.base_dir / "data" / "sp500"
        self.old_news = self.base_dir / "data" / "news"
        self.old_market = self.base_dir / "data" / "market"

        # New paths
        self.new_bronze_market = self.base_dir / "data_clean" / "bronze" / "market"
        self.new_bronze_news = self.base_dir / "data_clean" / "bronze" / "news"
        self.new_silver_market = self.base_dir / "data_clean" / "silver" / "market"
        self.new_silver_news = self.base_dir / "data_clean" / "silver" / "news"
        self.new_gold_market = self.base_dir / "data_clean" / "gold" / "market"
        self.new_gold_news = self.base_dir / "data_clean" / "gold" / "news"

    def create_structure(self):
        """Create the new directory structure."""
        logger.info("Creating new directory structure...")

        directories = [
            self.new_bronze_market,
            self.new_bronze_news,
            self.new_silver_market / "technical",
            self.new_silver_market / "microstructure",
            self.new_silver_market / "volatility",
            self.new_silver_news / "sentiment",
            self.new_silver_news / "entities",
            self.new_silver_news / "topics",
            self.new_gold_market / "features",
            self.new_gold_market / "labels",
            self.new_gold_news / "signals",
            self.base_dir / "data_clean" / "models",
            self.base_dir / "data_clean" / "training_outputs",
        ]

        for directory in directories:
            if not self.dry_run:
                directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"  {'[DRY RUN] Would create' if self.dry_run else 'Created'}: {directory}")

    def migrate_bronze_market(self):
        """Migrate bronze market data."""
        logger.info("\n" + "="*80)
        logger.info("MIGRATING BRONZE MARKET DATA")
        logger.info("="*80)

        source = self.old_bronze / "prices"
        if not source.exists():
            logger.warning(f"Source not found: {source}")
            return

        # Copy all NDJSON price files
        for file in source.glob("*.ndjson"):
            dest = self.new_bronze_market / file.name
            logger.info(f"  {file.name} -> {dest}")

            if not self.dry_run:
                shutil.copy2(file, dest)

    def migrate_bronze_news(self):
        """Migrate bronze news data."""
        logger.info("\n" + "="*80)
        logger.info("MIGRATING BRONZE NEWS DATA")
        logger.info("="*80)

        # Migrate from old news articles location
        source = self.old_bronze / "news" / "articles"
        if source.exists():
            for file in source.glob("*.json"):
                dest = self.new_bronze_news / file.name
                logger.info(f"  {file.name} -> {dest}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

        # Also check alternative location
        alt_source = self.old_news / "bronze" / "raw_articles"
        if alt_source.exists():
            for file in alt_source.glob("*.json"):
                dest = self.new_bronze_news / file.name
                if not dest.exists():  # Avoid duplicates
                    logger.info(f"  {file.name} -> {dest}")

                    if not self.dry_run:
                        shutil.copy2(file, dest)

    def migrate_silver_market(self):
        """Migrate silver market features."""
        logger.info("\n" + "="*80)
        logger.info("MIGRATING SILVER MARKET FEATURES")
        logger.info("="*80)

        # Technical features
        tech_source = self.old_sp500 / "silver" / "technical_features"
        if tech_source.exists():
            for file in tech_source.glob("*.csv"):
                dest = self.new_silver_market / "technical" / file.name
                logger.info(f"  technical: {file.name}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

        # Microstructure features
        micro_source = self.old_sp500 / "silver" / "microstructure"
        if micro_source.exists():
            for file in micro_source.glob("*.csv"):
                dest = self.new_silver_market / "microstructure" / file.name
                logger.info(f"  microstructure: {file.name}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

        # Volatility features
        vol_source = self.old_sp500 / "silver" / "volatility"
        if vol_source.exists():
            for file in vol_source.glob("*.csv"):
                dest = self.new_silver_market / "volatility" / file.name
                logger.info(f"  volatility: {file.name}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

    def migrate_silver_news(self):
        """Migrate silver news features."""
        logger.info("\n" + "="*80)
        logger.info("MIGRATING SILVER NEWS FEATURES")
        logger.info("="*80)

        # Sentiment scores
        sent_source = self.old_news / "silver" / "sentiment_scores"
        if sent_source.exists():
            for file in sent_source.glob("*.csv"):
                dest = self.new_silver_news / "sentiment" / file.name
                logger.info(f"  sentiment: {file.name}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

        # Entity mentions
        ent_source = self.old_news / "silver" / "entity_mentions"
        if ent_source.exists():
            for file in ent_source.glob("*.csv"):
                dest = self.new_silver_news / "entities" / file.name
                logger.info(f"  entities: {file.name}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

        # Topic signals
        topic_source = self.old_news / "silver" / "topic_signals"
        if topic_source.exists():
            for file in topic_source.glob("*.csv"):
                dest = self.new_silver_news / "topics" / file.name
                logger.info(f"  topics: {file.name}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

    def migrate_gold_market(self):
        """Migrate gold market features."""
        logger.info("\n" + "="*80)
        logger.info("MIGRATING GOLD MARKET FEATURES")
        logger.info("="*80)

        # Training features
        feat_source = self.old_sp500 / "gold" / "training"
        if feat_source.exists():
            for file in feat_source.glob("*.csv"):
                dest = self.new_gold_market / "features" / file.name
                logger.info(f"  features: {file.name}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

            for file in feat_source.glob("*.parquet"):
                dest = self.new_gold_market / "features" / file.name
                logger.info(f"  features: {file.name}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

    def migrate_gold_news(self):
        """Migrate gold news signals."""
        logger.info("\n" + "="*80)
        logger.info("MIGRATING GOLD NEWS SIGNALS")
        logger.info("="*80)

        # Trading signals
        signals_source = self.old_news / "gold" / "news_signals"
        if signals_source.exists():
            for file in signals_source.glob("*.csv"):
                dest = self.new_gold_news / "signals" / file.name
                logger.info(f"  signals: {file.name}")

                if not self.dry_run:
                    shutil.copy2(file, dest)

    def generate_readme(self):
        """Generate README for the new structure."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING STRUCTURE README")
        logger.info("="*80)

        readme_content = f"""# Clean Medallion Architecture - Data Structure

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure

```
data_clean/
├── bronze/              # Raw data (immutable)
│   ├── market/         # OHLCV candles from OANDA
│   └── news/           # Raw news articles (JSON)
│
├── silver/             # Processed features
│   ├── market/
│   │   ├── technical/      # RSI, MACD, Bollinger, etc.
│   │   ├── microstructure/ # Spread, volume, depth
│   │   └── volatility/     # GK, Parkinson, YZ estimators
│   └── news/
│       ├── sentiment/      # Sentiment scores
│       ├── entities/       # Entity mentions
│       └── topics/         # Topic classifications
│
├── gold/               # Training-ready data
│   ├── market/
│   │   ├── features/       # Merged market features
│   │   └── labels/         # Price prediction labels
│   └── news/
│       └── signals/        # Trading signals from news
│
├── models/             # Trained models
└── training_outputs/   # Training logs, metrics, plots
```

## Data Flow

### Market Data Pipeline
```
Bronze (NDJSON) → Silver (CSV) → Gold (CSV + Parquet)
  Raw candles   →   Features   →   Training data
```

### News Data Pipeline
```
Bronze (JSON) → Silver (CSV) → Gold (CSV)
 Raw articles →   Features   → Trading signals
```

## Timezone Standards

- **All timestamps in UTC**
- Market data: ISO 8601 format with 'Z' suffix
- News data: ISO 8601 format, normalized to UTC

## File Naming Conventions

### Bronze Layer
- Market: `{{instrument}}_{{granularity}}_{{years}}y_{{date}}.ndjson`
  Example: `spx500_usd_m1_5y_20251016.ndjson`

- News: `{{article_id}}.json`
  Example: `a1b2c3d4e5f6.json`

### Silver Layer
- Market: `{{instrument}}_{{feature_type}}_{{date}}.csv`
  Example: `spx500_technical_20251016.csv`

- News: `{{source}}_{{feature_type}}_{{date}}.csv`
  Example: `all_sources_sentiment_20251016.csv`

### Gold Layer
- Market: `{{instrument}}_features_{{date}}.csv`
  Example: `spx500_features_20251016.csv`

- Labels: `{{instrument}}_labels_{{horizon}}.csv`
  Example: `spx500_labels_30min.csv`

- News: `{{instrument}}_news_signals_{{date}}.csv`
  Example: `spx500_news_signals_20251016.csv`

## Data Quality Checks

1. **Timestamp Validation**: All timestamps must be valid UTC
2. **Completeness**: No missing required fields
3. **Deduplication**: Remove duplicate candles/articles
4. **Alignment**: Market and news data must overlap temporally

## Usage

See individual pipeline scripts in `src_clean/data_pipelines/`

"""

        readme_path = self.base_dir / "data_clean" / "README.md"

        if not self.dry_run:
            with open(readme_path, 'w') as f:
                f.write(readme_content)

        logger.info(f"  {'[DRY RUN] Would create' if self.dry_run else 'Created'}: {readme_path}")

    def create_migration_summary(self):
        """Create a migration summary report."""
        logger.info("\n" + "="*80)
        logger.info("MIGRATION SUMMARY")
        logger.info("="*80)

        summary = {
            "migration_date": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "source_locations": {
                "bronze": str(self.old_bronze),
                "sp500": str(self.old_sp500),
                "news": str(self.old_news),
                "market": str(self.old_market)
            },
            "destination": str(self.base_dir / "data_clean"),
            "structure_created": not self.dry_run
        }

        summary_path = self.base_dir / "data_clean" / "migration_summary.json"

        if not self.dry_run:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

        logger.info(f"  {'[DRY RUN] Would create' if self.dry_run else 'Created'}: {summary_path}")

    def run(self):
        """Execute the full migration."""
        logger.info("\n")
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*15 + "DATA MIGRATION TO CLEAN STRUCTURE" + " "*30 + "║")
        logger.info("╚" + "="*78 + "╝")
        logger.info("")

        if self.dry_run:
            logger.warning("DRY RUN MODE - No files will be moved")
            logger.info("")

        # Execute migration steps
        self.create_structure()
        self.migrate_bronze_market()
        self.migrate_bronze_news()
        self.migrate_silver_market()
        self.migrate_silver_news()
        self.migrate_gold_market()
        self.migrate_gold_news()
        self.generate_readme()
        self.create_migration_summary()

        logger.info("\n" + "="*80)
        if self.dry_run:
            logger.info("DRY RUN COMPLETE - No changes made")
            logger.info("Run with --execute to perform actual migration")
        else:
            logger.info("MIGRATION COMPLETE!")
            logger.info(f"New structure available at: {self.base_dir / 'data_clean'}")
        logger.info("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute migration (default is dry-run)"
    )

    args = parser.parse_args()

    migrator = DataMigrator(dry_run=not args.execute)
    migrator.run()


if __name__ == "__main__":
    main()
