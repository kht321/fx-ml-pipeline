# S&P 500 Historical Data Download

This document explains how to download 10 years of S&P 500 historical data from OANDA at 1-minute resolution for model training.

## Overview

The `download_sp500_historical.py` script downloads historical S&P 500 (SPX500_USD) candlestick data from OANDA and saves it in the Bronze layer for further processing.

### Key Features

- **Automatic Pagination**: Handles OANDA's 5,000 candle limit per request
- **Resume Capability**: Can resume interrupted downloads from where they left off
- **Rate Limiting**: Respects API rate limits with configurable delays
- **Progress Tracking**: Shows real-time progress and saves state periodically
- **Data Validation**: Built-in validation to check downloaded data integrity

## Prerequisites

1. **OANDA Account**: You need an OANDA account with API access
2. **API Credentials**: Set up your `.env` file with:
   ```
   OANDA_TOKEN=your_api_token_here
   OANDA_ENV=practice  # or 'live'
   ```
3. **Python Environment**: Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Important Limitations

⚠️ **Data Availability**: OANDA typically provides:
- **1-minute data**: ~1-2 years of history (varies by instrument)
- **Hourly data**: ~4-5 years of history
- **Daily data**: 10+ years of history

For 10 years of training data at high resolution, you may need to:
1. Start with whatever 1-minute data is available (likely 1-2 years)
2. Use hourly data (H1) for the full 10 years
3. Or combine both: recent 1-minute + older hourly data

## Usage

### Basic Usage (10 years of 1-minute data)

```bash
cd "/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline"
source .venv/bin/activate
python src/download_sp500_historical.py
```

This will:
- Download up to 10 years of S&P 500 1-minute candles (subject to OANDA's data availability)
- Save to: `data/bronze/prices/spx500_usd_m1_historical.ndjson`
- Show progress and estimated completion time

### Download Hourly Data Instead (Better for 10 years)

```bash
python src/download_sp500_historical.py --granularity H1
```

This will:
- Download 10 years of hourly S&P 500 candles (more likely to be available)
- Save to: `data/bronze/prices/spx500_usd_h1_historical.ndjson`

### Other Options

```bash
# Download only 5 years
python src/download_sp500_historical.py --years 5

# Download with slower rate limiting (if hitting API limits)
python src/download_sp500_historical.py --rate-limit-delay 1.0

# Change output directory
python src/download_sp500_historical.py --output-dir data/custom_path

# Validate existing data without downloading
python src/download_sp500_historical.py --validate-only
```

### Resume Interrupted Download

If the download is interrupted (Ctrl+C, network issue, etc.), simply run the same command again:

```bash
python src/download_sp500_historical.py
```

The script will automatically resume from where it left off using the progress file:
`data/bronze/prices/spx500_usd_m1_progress.json`

## Output Format

Data is saved in **NDJSON** (Newline Delimited JSON) format. Each line is a complete JSON object representing one candle:

```json
{
  "time": "2024-01-15T14:30:00.000000000Z",
  "instrument": "SPX500_USD",
  "granularity": "M1",
  "open": 4780.25,
  "high": 4782.50,
  "low": 4779.75,
  "close": 4781.00,
  "volume": 1234,
  "collected_at": "2024-10-12T10:30:00.000000Z"
}
```

### Data Fields

- `time`: Candle timestamp (UTC)
- `instrument`: Trading instrument (SPX500_USD)
- `granularity`: Time interval (M1 = 1 minute)
- `open`, `high`, `low`, `close`: OHLC prices (mid prices)
- `volume`: Trading volume
- `collected_at`: When the data was downloaded

## Expected Download Time

For a full 10-year download at 1-minute resolution:

- **Estimated candles**: ~2.6 million (assuming market hours)
- **API calls needed**: ~520 requests (5,000 candles each)
- **Estimated time**: 4-8 minutes (with 0.5s delay between requests)
- **File size**: ~200-300 MB

**Note**: Actual time depends on:
- API response times
- Rate limiting settings
- Network speed
- Data availability

## Monitoring Progress

The script provides detailed logging:

```
2024-10-12 10:30:00 - INFO - Starting download of 10 years of SPX500_USD M1 data
2024-10-12 10:30:00 - INFO - Output file: data/bronze/prices/spx500_usd_m1_historical.ndjson
2024-10-12 10:30:00 - INFO - Estimated chunks to download: 520
2024-10-12 10:30:05 - INFO - Fetching chunk 1: 2014-10-12 10:30:00 to 2014-10-15 20:10:00
2024-10-12 10:30:06 - INFO - Saved 4823 candles. Total: 4823 | Progress: 1/520 chunks
2024-10-12 10:30:06 - INFO - Progress: 0.19%
...
```

## Next Steps After Download

Once data is downloaded, you can:

1. **Validate the data**:
   ```bash
   python src/download_sp500_historical.py --validate-only
   ```

2. **Process into features**: Use the existing pipeline scripts to transform Bronze data into Silver/Gold layers

3. **Combine with existing data**: Merge with USD_SGD or other instrument data for multi-asset training

4. **Train models**: Use the downloaded data for model training

## Troubleshooting

### Problem: "No candles returned"

**Cause**: OANDA may not have data for the requested time period.

**Solution**:
- Try a shorter time period: `--years 2`
- Use hourly data instead: `--granularity H1`
- Check OANDA's data availability documentation

### Problem: "Rate limit exceeded"

**Cause**: Making too many API requests too quickly.

**Solution**: Increase the delay between requests:
```bash
python src/download_sp500_historical.py --rate-limit-delay 2.0
```

### Problem: "Connection timeout"

**Cause**: Network issues or OANDA API downtime.

**Solution**:
- Check your internet connection
- Check OANDA API status
- Resume the download later (progress is saved)

### Problem: "ModuleNotFoundError"

**Cause**: Virtual environment not activated or dependencies not installed.

**Solution**:
```bash
source .venv/bin/activate
pip install -r requirements.txt  # if requirements file exists
```

## Data Quality Notes

- Only **complete** candles are saved (incomplete real-time candles are filtered out)
- Timestamps are in UTC
- Prices are **mid prices** (average of bid/ask)
- Weekends and holidays may have no data (markets closed)
- Volume may be 0 for some candles depending on OANDA's data

## Performance Tips

1. **Run overnight**: Large downloads can take hours
2. **Use screen/tmux**: Prevent interruption from terminal disconnection
3. **Monitor disk space**: 10 years of 1-minute data can be large
4. **Start with recent data**: Download recent years first for immediate use

## Example: Download Strategy

For maximum data coverage, consider this strategy:

```bash
# 1. Download recent 1-minute data (likely 1-2 years available)
python src/download_sp500_historical.py --years 2 --granularity M1

# 2. Download older hourly data (for full 10 years)
python src/download_sp500_historical.py --years 10 --granularity H1

# 3. Validate both datasets
python src/download_sp500_historical.py --granularity M1 --validate-only
python src/download_sp500_historical.py --granularity H1 --validate-only
```

This gives you:
- High-resolution recent data (1-minute)
- Long history at lower resolution (hourly)
- Best of both worlds for training

## Support

For issues with:
- **OANDA API**: Check OANDA documentation or support
- **Script errors**: Check logs and error messages
- **Data quality**: Use the `--validate-only` flag

## Files Created

The script creates these files in `data/bronze/prices/`:

1. `spx500_usd_m1_historical.ndjson` - The actual data
2. `spx500_usd_m1_progress.json` - Resume progress (deleted after completion)

Both files are automatically created with appropriate permissions.
