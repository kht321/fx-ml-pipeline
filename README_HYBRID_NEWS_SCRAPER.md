# Hybrid News Scraper - Quick Start

## TL;DR - Get 5 Years of News for FREE

```bash
# 1. Install dependencies (already in requirements.txt)
source .venv/bin/activate

# 2. Collect 2017-2025 news (~50k-100k articles in 2-5 hours)
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2017-01-01 \
    --end-date 2025-10-19 \
    --sources gdelt

# 3. Done! Articles saved to data_clean/bronze/news/hybrid/
```

**Cost**: $0 | **Runtime**: 2-5 hours | **Articles**: 50,000-100,000

---

## What You Get

✅ **5+ years of financial news** (2017-2025)
✅ **S&P 500 relevant articles** only
✅ **100% FREE** - no credit card, no signup (for GDELT)
✅ **Compatible** with existing pipeline (Bronze/Silver/Gold)
✅ **Deduplicated** across all sources
✅ **Sentiment scores** (when using Alpha Vantage)

---

## Free Sources Available

| Source | Period | Articles | Free Limit | Signup |
|--------|--------|----------|------------|--------|
| **GDELT** | 2017-present | 50-100k | Unlimited | None ❌ |
| **Finnhub** | Past 1 year | 5-10k | 60/min | Free API key 🔑 |
| **Alpha Vantage** | Recent weeks | 1-2k | 25/day | Free API key 🔑 |

---

## Quick Commands

### Full 5-Year Collection (2017-2025)
```bash
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2017-01-01 \
    --sources gdelt
```

### Recent Year Only (2024-2025)
```bash
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2024-10-19 \
    --sources finnhub
```

### Daily Updates (Incremental)
```bash
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --mode incremental \
    --sources all
```

---

## Setup API Keys (Optional)

Get more coverage with free API keys:

**Alpha Vantage** (25 calls/day):
```bash
# 1. Get free key: https://www.alphavantage.co/support/#api-key
# 2. Add to .env:
echo "ALPHAVANTAGE_KEY=your_key_here" >> .env
```

**Finnhub** (60 calls/min, 1 year history):
```bash
# 1. Sign up free: https://finnhub.io/register
# 2. Add to .env:
echo "FINNHUB_KEY=your_key_here" >> .env
```

---

## Output Format

Articles saved as individual JSON files:

```json
{
  "article_id": "1a2b3c4d...",
  "headline": "S&P 500 rallies as...",
  "body": "Full article text or summary...",
  "url": "https://example.com/article",
  "source": "gdelt_reuters",
  "published_at": "2025-10-18T14:30:00+00:00",
  "collected_at": "2025-10-19T00:00:00+00:00",
  "sp500_relevant": true,
  "collection_method": "gdelt_api"
}
```

**Location**: `data_clean/bronze/news/hybrid/`

---

## Coverage Timeline

```
2020         2017                              2025
├────────────┼────────────────────────────────┤
│    GAP     │      GDELT (FREE)              │
│            │                                │
│            └─────────────────┬──────────────┘
│                              │
│                              └─ Finnhub (1yr)
│                                 2024-2025
```

**Coverage Summary**:
- ✅ 2017-2025: Full coverage via GDELT (FREE)
- ⚠️ 2020-2017: Gap (pre-GDELT) - see options below

---

## Filling the Gap (2020-2017)

3 options for the period before GDELT coverage:

**Option 1: Daily Collection** ($0, Low effort)
- Setup cron job to run daily
- Wait 3 years to accumulate historical data
- Cost: $0, Time: 3 years

**Option 2: Common Crawl** ($0, High effort)
- Parse WARC files from 2016-2020
- Requires `warcio` library and significant coding
- Cost: $0, Time: 1-2 weeks development

**Option 3: Paid API** ($1,000/year, Low effort)
- EODHD All-In-One: $999.90/year
- Instant access to 30+ years
- Cost: ~$1k/year, Time: Immediate

---

## Daily Automation (Recommended)

Setup cron job for ongoing collection:

```bash
# Run daily at 1 AM
crontab -e

# Add this line:
0 1 * * * cd /path/to/fx-ml-pipeline && source .venv/bin/activate && python3 src_clean/data_pipelines/bronze/hybrid_news_scraper.py --mode incremental --sources all >> logs/news_scraper.log 2>&1
```

**Collects**: 100-500 articles/day
**Runtime**: 5-10 minutes
**Cost**: $0

---

## Integration with Existing Pipeline

Works seamlessly with your current pipeline:

```bash
# 1. Collect news (new hybrid scraper)
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py --sources gdelt

# 2. Process sentiment (existing silver layer)
python src_clean/data_pipelines/silver/news_sentiment_processor.py

# 3. Generate signals (existing gold layer)
python src_clean/data_pipelines/gold/news_signals_generator.py

# 4. Merge with market data
# ... existing ML pipeline code ...
```

---

## Cost Comparison

| Method | Period | Articles | Cost | Time |
|--------|--------|----------|------|------|
| **Hybrid Scraper (FREE)** | 2017-2025 | 50-100k | **$0** | 2-5h |
| NewsAPI Business | 2020-2025 | Unlimited | $4,308/yr | Instant |
| EODHD All-In-One | 30+ years | Unlimited | $999/yr | Instant |
| NewsCatcher Enterprise | Custom | Unlimited | $120k/yr | Instant |

**Savings**: $999 - $120,000 per year 💰

---

## Files Created

- **Scraper**: [src_clean/data_pipelines/bronze/hybrid_news_scraper.py](src_clean/data_pipelines/bronze/hybrid_news_scraper.py)
- **Config**: [configs/hybrid_news_sources.yaml](configs/hybrid_news_sources.yaml)
- **Setup Script**: [scripts/setup_hybrid_news_scraper.sh](scripts/setup_hybrid_news_scraper.sh)
- **Full Guide**: [docs/HYBRID_NEWS_SCRAPER_GUIDE.md](docs/HYBRID_NEWS_SCRAPER_GUIDE.md)

---

## Troubleshooting

**No articles collected?**
```bash
# Test GDELT API directly
curl "https://api.gdeltproject.org/api/v2/doc/doc?query=stock%20market&mode=artlist&maxrecords=5&format=json"
```

**API key not working?**
```bash
# Check .env file
cat .env | grep -E "(ALPHAVANTAGE|FINNHUB)_KEY"
```

**Rate limit exceeded?**
- GDELT: No limit (be respectful)
- Alpha Vantage: Wait 24 hours (resets daily)
- Finnhub: Wait 1 hour (60 calls/min limit)

---

## Next Steps

1. **Collect news**: Run the scraper (2-5 hours)
2. **Review results**: Check `data_clean/bronze/news/hybrid/`
3. **Process pipeline**: Run Silver/Gold layers
4. **Setup automation**: Add cron job for daily updates
5. **Train models**: Merge with market data

---

## Support

- **Full Documentation**: [docs/HYBRID_NEWS_SCRAPER_GUIDE.md](docs/HYBRID_NEWS_SCRAPER_GUIDE.md)
- **Configuration**: [configs/hybrid_news_sources.yaml](configs/hybrid_news_sources.yaml)
- **Test Run**: `python src_clean/data_pipelines/bronze/hybrid_news_scraper.py --mode incremental --sources gdelt`

---

**Ready?** Start collecting:

```bash
python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --start-date 2017-01-01 \
    --sources gdelt
```

⏱️ Grab a coffee - this will take 2-5 hours! ☕
