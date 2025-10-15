# Data Download Instructions

## Why Data Files Are Not in Git

Large data files (>100MB) are excluded from git to keep the repository lightweight. You need to generate them locally.

## Files Excluded

- `data_clean/bronze/market/*.ndjson` - Raw market data (353MB+)
- `data_clean/silver/**/*.csv` - Processed features (600MB+)
- `data_clean/gold/**/*.csv` - Training data (900MB+)
- `outputs/` - Generated reports and features
- `data/sp500/` - Old data structure

## How to Get the Data

### Option 1: Use Existing Data (If Already Downloaded)

If you already have `data/bronze/prices/spx500_usd_m1_5years.ndjson`:

```bash
# Migrate existing data to clean structure
python migrate_to_clean_structure.py --execute

# Run pipeline to generate features
python src_clean/run_full_pipeline.py \
  --bronze-market data_clean/bronze/market/spx500_usd_m1_5years.ndjson \
  --skip-news \
  --prediction-horizon 30
```

### Option 2: Download Fresh Market Data

```bash
# 1. Set up environment variables in .env
echo "OANDA_TOKEN=your_token_here" >> .env
echo "OANDA_ACCOUNT_ID=your_account_id" >> .env

# 2. Download 5 years of S&P 500 data
python src_clean/data_pipelines/bronze/market_data_downloader.py \
  --years 5 \
  --instrument SPX500_USD \
  --granularity M1 \
  --output-dir data_clean/bronze/market

# 3. Run pipeline to generate features
python src_clean/run_full_pipeline.py \
  --bronze-market data_clean/bronze/market/spx500_usd_m1_5y_*.ndjson \
  --skip-news \
  --prediction-horizon 30
```

**Expected time**:
- Download: 30-60 minutes (depends on API speed)
- Pipeline: 2-3 minutes (processes 1.7M candles)

### Option 3: Request Data from Team

If someone on your team has already downloaded the data:

1. Ask them to share `data/bronze/prices/spx500_usd_m1_5years.ndjson`
2. Place it in your `data/bronze/prices/` directory
3. Run migration and pipeline (see Option 1)

## Data Size Reference

| File | Size | Description |
|------|------|-------------|
| `spx500_usd_m1_5years.ndjson` | 353 MB | Raw 5-year 1-min candles |
| `spx500_technical.csv` | 630 MB | Technical features |
| `spx500_microstructure.csv` | 701 MB | Microstructure features |
| `spx500_volatility.csv` | 884 MB | Volatility features |
| `spx500_features.csv` | 901 MB | Merged gold features |

**Total**: ~3.5 GB of generated data

## What's Included in Git

- ✅ All source code (`src_clean/`)
- ✅ Configuration files
- ✅ Documentation (`docs/`)
- ✅ Small sample data for testing
- ✅ News articles (JSON, small files)

## .gitignore Configuration

The `.gitignore` file excludes:
```
data_clean/bronze/market/*.ndjson
data_clean/silver/**/*.csv
data_clean/gold/**/*.csv
outputs/
*.csv
*.pkl
```

## Troubleshooting

**Q: Git push says files are too large**
A: Make sure `.gitignore` is properly configured and run:
```bash
git rm --cached -r data_clean/bronze/market/ data_clean/silver/ data_clean/gold/
git commit -m "Remove large data files from git"
git push
```

**Q: Pipeline fails - file not found**
A: You need to generate the data first (see options above)

**Q: Can I use Git LFS instead?**
A: Yes, but it's not recommended for ML projects. Better to:
- Store data in cloud storage (S3, GCS, Azure Blob)
- Download via scripts
- Keep git repo lightweight

## Recommended Workflow

1. Clone repo (small, fast)
2. Download/generate data locally (one-time, 30-60 min)
3. Run pipeline when needed (2-3 minutes)
4. Commit code changes only (fast git operations)

---

**Note**: Data files should be treated as artifacts, not source code. Generate them locally or download from shared storage.
