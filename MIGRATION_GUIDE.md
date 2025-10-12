# Migration Guide: USD/SGD → S&P 500 Data

This guide explains the changes made to switch from USD/SGD forex data to S&P 500 index data for model training.

## Summary of Changes

| Aspect | Old (USD/SGD) | New (S&P 500) |
|--------|---------------|---------------|
| **Instrument** | USD_SGD | SPX500_USD |
| **Timeframe** | Jan - Oct 2025 | 10 years historical |
| **Resolution** | 1 hour (H1) | 1 minute (M1) |
| **Script** | `hourly_candle_collector.py` | `download_sp500_historical.py` |
| **Data Source** | Live streaming | Historical backfill |
| **Use Case** | Real-time 2025 data | Training dataset |

## What Changed

### 1. **New Download Script**

**File**: [`src/download_sp500_historical.py`](src/download_sp500_historical.py)

A comprehensive script to download S&P 500 historical data with:
- Automatic pagination for large datasets
- Resume capability for interrupted downloads
- Progress tracking and logging
- Rate limiting
- Data validation

### 2. **Convenience Shell Script**

**File**: [`scripts/download_sp500_data.sh`](scripts/download_sp500_data.sh)

A bash wrapper for easy execution:
```bash
./scripts/download_sp500_data.sh --hourly  # For 10 years of hourly data
```

### 3. **Documentation**

**File**: [`src/README_SP500_DOWNLOAD.md`](src/README_SP500_DOWNLOAD.md)

Complete guide covering:
- Usage instructions
- Configuration options
- Troubleshooting
- Data format details
- Best practices

## Original vs New Approach

### Original: USD_SGD Hourly Collector

```python
# hourly_candle_collector.py
collector = HourlyCandleCollector(
    instrument="USD_SGD",
    output_dir="data/bronze/prices"
)
collector.start_live_collection()
```

**Characteristics:**
- ✓ Streams live data as it happens
- ✓ Good for real-time trading applications
- ✗ Only collects current year (2025)
- ✗ Limited historical data
- ✗ Hourly resolution only

### New: S&P 500 Historical Downloader

```python
# download_sp500_historical.py
downloader = SP500HistoricalDownloader(
    instrument="SPX500_USD",
    granularity="M1",  # 1-minute candles
    years_back=10
)
downloader.download_historical_data()
```

**Characteristics:**
- ✓ Downloads up to 10 years of historical data
- ✓ 1-minute resolution (60× more granular)
- ✓ Resume capability
- ✓ Progress tracking
- ✓ Better for model training with more data
- ⚠️ Limited by OANDA's historical data availability

## Data Format Comparison

Both scripts output data in the same Bronze layer NDJSON format:

### Common Fields
```json
{
  "time": "2024-01-15T14:30:00.000000000Z",
  "instrument": "SPX500_USD",  // was "USD_SGD"
  "granularity": "M1",          // was "H1"
  "open": 4780.25,
  "high": 4782.50,
  "low": 4779.75,
  "close": 4781.00,
  "volume": 1234,
  "collected_at": "2024-10-12T10:30:00.000000Z"
}
```

### Differences

**USD_SGD (old)** also included:
```json
{
  "bid_open": 1.3401,
  "bid_high": 1.3405,
  "bid_low": 1.3399,
  "bid_close": 1.3402,
  "ask_open": 1.3403,
  "ask_high": 1.3407,
  "ask_low": 1.3401,
  "ask_close": 1.3404,
  "spread": 0.0002
}
```

**Note**: The S&P 500 script can be modified to include bid/ask data by changing the `price` parameter from `"M"` (mid only) to `"MBA"` (mid, bid, ask).

## How to Use the New System

### Option 1: Download 10 Years of Hourly Data (Recommended)

Most reliable for getting full 10 years:

```bash
cd "/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline"
source .venv/bin/activate

# Using Python directly
python src/download_sp500_historical.py --granularity H1 --years 10

# OR using the convenience script
./scripts/download_sp500_data.sh --hourly
```

**Expected Output:**
- File: `data/bronze/prices/spx500_usd_h1_historical.ndjson`
- Candles: ~87,600 (10 years × 365 days × 24 hours)
- Time: ~5-10 minutes

### Option 2: Download 1-2 Years of 1-Minute Data

For highest resolution on recent data:

```bash
# Using Python
python src/download_sp500_historical.py --granularity M1 --years 2

# OR using the convenience script
./scripts/download_sp500_data.sh --years 2
```

**Expected Output:**
- File: `data/bronze/prices/spx500_usd_m1_historical.ndjson`
- Candles: ~500,000+ (limited by OANDA's availability)
- Time: ~10-30 minutes

### Option 3: Hybrid Approach (Best of Both Worlds)

Download both and use them for different purposes:

```bash
# Recent high-resolution data
python src/download_sp500_historical.py --granularity M1 --years 2

# Full historical context at lower resolution
python src/download_sp500_historical.py --granularity H1 --years 10
```

Then in your training pipeline, you can:
- Use M1 data for fine-grained pattern recognition
- Use H1 data for longer-term trends and seasonality
- Combine both for multi-timeframe analysis

## Impact on Existing Pipeline

### Components That Need Updates

1. **Feature Engineering** (`build_market_features.py`)
   - Update to handle S&P 500 price ranges
   - May need to adjust technical indicator parameters

2. **Labels** (`build_labels.py`)
   - Update return calculations for equity index vs forex
   - Adjust thresholds for S&P 500 volatility

3. **Model Training** (`train_combined_model.py`)
   - Update data loading paths
   - May need to retrain with new instrument characteristics

### Components That Work As-Is

These should work without changes:
- ✓ `oanda_api.py` - Already supports any OANDA instrument
- ✓ `fetch_candles.py` - Generic candle fetcher
- ✓ Bronze layer structure - Same NDJSON format
- ✓ OANDA authentication - No changes needed

## Key Differences: Forex vs Equity Index

### USD/SGD (Forex Pair)
- **Trading hours**: 24/5 (24 hours, 5 days/week)
- **Volatility**: Lower, more stable
- **Typical daily range**: 0.1% - 0.5%
- **Bid/Ask spread**: Very tight (< 0.01%)
- **Drivers**: Interest rates, economic data, central bank policy

### SPX500_USD (S&P 500 CFD)
- **Trading hours**: 24/5 with gaps during US market close
- **Volatility**: Higher, especially during market hours
- **Typical daily range**: 0.5% - 2.0%
- **Bid/Ask spread**: Wider than forex
- **Drivers**: Corporate earnings, economic data, sentiment, liquidity

### Implications for Models

You may need to:
1. **Adjust volatility thresholds**: S&P 500 moves more than forex
2. **Account for market hours**: Gaps in overnight trading
3. **Update feature scaling**: Different price ranges and distributions
4. **Retrain models**: New instrument = new patterns
5. **Review risk parameters**: Different volatility characteristics

## Next Steps

1. **Download the data**:
   ```bash
   ./scripts/download_sp500_data.sh --hourly
   ```

2. **Validate the download**:
   ```bash
   python src/download_sp500_historical.py --validate-only
   ```

3. **Explore the data**:
   ```python
   import json
   with open('data/bronze/prices/spx500_usd_h1_historical.ndjson') as f:
       for i, line in enumerate(f):
           if i < 5:  # Print first 5 candles
               print(json.loads(line))
   ```

4. **Update feature engineering**:
   - Modify scripts in `src/` to use new instrument
   - Adjust parameters for S&P 500 characteristics

5. **Retrain models**:
   - Use the new historical data
   - Validate performance
   - Compare with USD/SGD results

## Keeping the Old System

The original `hourly_candle_collector.py` has **not been modified**. You can still use it for:
- Collecting live USD/SGD data
- Real-time trading applications
- Comparing forex vs equity models

Both systems can coexist in the same pipeline.

## Troubleshooting

### Issue: "No candles returned"

The requested time period may exceed OANDA's data availability.

**Solution**: Reduce `--years` or use `--granularity H1`

### Issue: "Rate limit exceeded"

Too many requests too quickly.

**Solution**: Increase `--rate-limit-delay 2.0`

### Issue: Download interrupted

**Solution**: Just run the script again - it will resume from where it stopped

## References

- [OANDA v20 API Documentation](https://developer.oanda.com/rest-live-v20/introduction/)
- [OANDA Instrument List](https://www.oanda.com/us-en/trading/instruments/)
- Original script: [`src/hourly_candle_collector.py`](src/hourly_candle_collector.py)
- New script: [`src/download_sp500_historical.py`](src/download_sp500_historical.py)

---

**Need Help?**

Check the detailed README: [`src/README_SP500_DOWNLOAD.md`](src/README_SP500_DOWNLOAD.md)
