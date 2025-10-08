# Complete ETL Documentation: FX ML Pipeline

This document provides comprehensive ETL (Extract-Transform-Load) descriptions for **all transformation stages** in both the Market and News medallion pipelines.

---

## Table of Contents

1. [Market Data Medallion](#market-data-medallion)
   - [ETL 1: API → Bronze](#market-etl-1-api--bronze)
   - [ETL 2: Bronze → Silver](#market-etl-2-bronze--silver)
   - [ETL 3: Silver → Gold](#market-etl-3-silver--gold)

2. [News Data Medallion](#news-data-medallion)
   - [ETL 1: Web Scraping → Bronze](#news-etl-1-web-scraping--bronze)
   - [ETL 2: Bronze → Silver (FinGPT)](#news-etl-2-bronze--silver-fingpt)
   - [ETL 3: Silver → Gold](#news-etl-3-silver--gold)

3. [Combined Training Layer](#combined-training-layer)
   - [ETL 4: Gold Layers → Combined Training](#combined-etl-gold-layers--combined-training)

---

# Market Data Medallion

## Market ETL 1: API → Bronze

### **Extract Phase**

**Source**: OANDA v20 REST API
**Endpoint**: `/v3/instruments/{instrument}/candles`
**Implementation**: [`hourly_candle_collector.py`](src/hourly_candle_collector.py)

```python
# API Request
GET https://api-fxpractice.oanda.com/v3/instruments/USD_SGD/candles
?granularity=H1
&count=500
&price=MBA  # Mid, Bid, Ask prices
```

**API Response Schema**:
```json
{
  "instrument": "USD_SGD",
  "granularity": "H1",
  "candles": [
    {
      "complete": true,
      "time": "2025-01-15T08:00:00.000000Z",
      "volume": 1234,
      "mid": {
        "o": "1.3450",
        "h": "1.3460",
        "l": "1.3445",
        "c": "1.3452"
      },
      "bid": {
        "o": "1.3449",
        "h": "1.3459",
        "l": "1.3444",
        "c": "1.3451"
      },
      "ask": {
        "o": "1.3451",
        "h": "1.3461",
        "l": "1.3446",
        "c": "1.3453"
      }
    }
  ]
}
```

### **Transform Phase**

**Transformations**: Minimal (format conversion only)

```python
def transform_candle_to_bronze(api_candle):
    """Transform OANDA API response to Bronze format"""

    # Extract numeric values from string prices
    bronze_candle = {
        "time": api_candle["time"],
        "instrument": "USD_SGD",
        "granularity": "H1",

        # Mid prices (type conversion only)
        "open": float(api_candle["mid"]["o"]),
        "high": float(api_candle["mid"]["h"]),
        "low": float(api_candle["mid"]["l"]),
        "close": float(api_candle["mid"]["c"]),
        "volume": int(api_candle["volume"]),

        # Bid prices
        "bid_open": float(api_candle["bid"]["o"]),
        "bid_high": float(api_candle["bid"]["h"]),
        "bid_low": float(api_candle["bid"]["l"]),
        "bid_close": float(api_candle["bid"]["c"]),

        # Ask prices
        "ask_open": float(api_candle["ask"]["o"]),
        "ask_high": float(api_candle["ask"]["h"]),
        "ask_low": float(api_candle["ask"]["l"]),
        "ask_close": float(api_candle["ask"]["c"]),

        # Derived: Spread calculation
        "spread": float(api_candle["ask"]["c"]) - float(api_candle["bid"]["c"]),

        # Metadata
        "collected_at": datetime.utcnow().isoformat() + "Z"
    }

    return bronze_candle
```

**Transformations Applied**:
1. ✅ Type conversion (string → float/int)
2. ✅ Spread calculation (ask_close - bid_close)
3. ✅ Metadata addition (collected_at timestamp)
4. ❌ No filtering, aggregation, or feature engineering

### **Load Phase**

**Destination**: `data/bronze/prices/usd_sgd_hourly_2025.ndjson`
**Format**: NDJSON (Newline-Delimited JSON)
**Write Mode**: Append-only streaming

```python
def save_candle_to_bronze(candle, output_file):
    """Append candle to NDJSON file"""
    with open(output_file, 'a') as f:
        f.write(json.dumps(candle) + '\n')
        f.flush()  # Ensure immediate write (crash recovery)
```

**Output Example** (one line per candle):
```json
{"time":"2025-01-15T08:00:00.000000Z","instrument":"USD_SGD","granularity":"H1","open":1.3450,"high":1.3460,"low":1.3445,"close":1.3452,"volume":1234,"bid_open":1.3449,"bid_high":1.3459,"bid_low":1.3444,"bid_close":1.3451,"ask_open":1.3451,"ask_high":1.3461,"ask_low":1.3446,"ask_close":1.3453,"spread":0.0002,"collected_at":"2025-01-15T08:05:23.123456Z"}
```

### **Data Quality Checks**

```python
def quality_checks(candle):
    """Bronze layer quality validation"""

    # Only save complete candles
    if not candle.get("complete", False):
        return False

    # Ensure spread is positive
    if candle["spread"] < 0:
        logger.warning(f"Negative spread detected: {candle}")
        return False

    # Ensure OHLC consistency
    if not (candle["low"] <= candle["close"] <= candle["high"]):
        logger.warning(f"OHLC inconsistency: {candle}")
        return False

    return True
```

### **Deduplication Strategy**

```python
def get_latest_candle_time(output_file):
    """Check last collected candle to avoid duplicates"""
    if not os.path.exists(output_file):
        return None

    with open(output_file, 'r') as f:
        lines = f.readlines()
        if lines:
            last_candle = json.loads(lines[-1])
            return datetime.fromisoformat(last_candle['time'])

    return None

def fetch_new_candles_only(last_collected_time):
    """Fetch only candles after last collected timestamp"""
    params = {
        "granularity": "H1",
        "from": (last_collected_time + timedelta(hours=1)).isoformat()
    }
    return oanda_api.fetch_candles(params)
```

### **ETL Summary: API → Bronze**

| Aspect | Details |
|--------|---------|
| **Frequency** | Every 10 minutes (check for new hourly candles) |
| **Latency** | <1 second per candle |
| **Data Volume** | ~24 candles/day × 365 days = 8,760 candles/year |
| **File Size** | ~350 bytes/candle × 8,760 = ~3 MB/year |
| **Transformations** | Type conversion + spread calculation only |
| **Validation** | Complete candles only, positive spread, OHLC consistency |

---

## Market ETL 2: Bronze → Silver

### **Extract Phase**

**Source**: `data/bronze/prices/usd_sgd_hourly_2025.ndjson`
**Implementation**: [`build_market_features.py`](src/build_market_features.py)

```python
def extract_from_bronze(bronze_file):
    """Read NDJSON and parse into DataFrame"""
    candles = []

    with open(bronze_file, 'r') as f:
        for line in f:
            candle = json.loads(line.strip())

            # Extract price tick data
            tick = {
                "time": pd.to_datetime(candle["time"]),
                "instrument": candle["instrument"],
                "best_bid": candle["bid_close"],
                "best_ask": candle["ask_close"],
                "mid": (candle["bid_close"] + candle["ask_close"]) / 2,
                "spread": candle["spread"],
                "bid_liquidity": candle.get("bid_liquidity", None),
                "ask_liquidity": candle.get("ask_liquidity", None),
                "volume": candle["volume"]
            }

            candles.append(tick)

    return pd.DataFrame(candles)
```

### **Transform Phase**

**Transformations**: Feature engineering split into 3 parallel streams

#### **T1: Technical Features**

**Purpose**: Price-based indicators for trend and momentum
**Dependencies**: Only `mid` price and `spread`
**Output**: `data/market/silver/technical_features/sgd_vs_majors.csv`

```python
def compute_technical_features(df):
    """Generate technical indicators from price data"""

    df = df.sort_values('time').copy()

    # Returns (price changes)
    df['ret_1'] = df['mid'].pct_change(1)      # 1-period return
    df['ret_5'] = df['mid'].pct_change(5)      # 5-period return

    # Rolling statistics (20-period window)
    df['roll_vol_20'] = df['ret_1'].rolling(20, min_periods=5).std()
    df['roll_mean_20'] = df['mid'].rolling(20, min_periods=5).mean()

    # Z-score (price deviation from mean)
    df['zscore_20'] = (df['mid'] - df['roll_mean_20']) / df['roll_vol_20']

    # Exponential moving averages
    df['ewma_short'] = df['mid'].ewm(span=5, adjust=False).mean()
    df['ewma_long'] = df['mid'].ewm(span=20, adjust=False).mean()
    df['ewma_signal'] = df['ewma_short'] - df['ewma_long']  # Crossover signal

    # Spread analysis
    df['spread_pct'] = df['spread'] / df['mid']  # Spread as % of price
    df['spread_zscore'] = (
        (df['spread'] - df['spread'].rolling(20).mean()) /
        df['spread'].rolling(20).std()
    )

    return df
```

**Features Created** (9 features):
- `ret_1`, `ret_5`: Returns
- `roll_vol_20`, `roll_mean_20`: Rolling statistics
- `zscore_20`: Price deviation
- `ewma_short`, `ewma_long`, `ewma_signal`: Moving averages
- `spread_pct`, `spread_zscore`: Spread metrics

#### **T2: Microstructure Features**

**Purpose**: Liquidity and order book depth metrics
**Dependencies**: `bid_liquidity`, `ask_liquidity`, `spread`
**Output**: `data/market/silver/microstructure/depth_features.csv`

```python
def compute_microstructure_features(df):
    """Generate market microstructure features"""

    df = df.sort_values('time').copy()

    # Liquidity metrics
    df['total_liquidity'] = df['bid_liquidity'] + df['ask_liquidity']
    df['liquidity_imbalance'] = (
        (df['ask_liquidity'] - df['bid_liquidity']) / df['total_liquidity']
    )

    # Bid-ask spread analysis
    df['effective_spread'] = df['spread'] / df['mid']
    df['quoted_depth'] = np.minimum(df['bid_liquidity'], df['ask_liquidity'])

    # Rolling liquidity statistics
    df['avg_liquidity_20'] = df['total_liquidity'].rolling(20, min_periods=5).mean()
    df['liquidity_shock'] = (
        (df['total_liquidity'] - df['avg_liquidity_20']) / df['avg_liquidity_20']
    )

    return df
```

**Features Created** (6 features):
- `total_liquidity`, `liquidity_imbalance`: Liquidity metrics
- `effective_spread`, `quoted_depth`: Trading cost metrics
- `avg_liquidity_20`, `liquidity_shock`: Liquidity dynamics

**Special Handling**: Sparse data (order book not always available)
```python
# If no order book data, microstructure CSV will have fewer rows
# Gold layer will handle missing values via forward-fill
```

#### **T3: Volatility Features**

**Purpose**: Risk and regime detection metrics
**Dependencies**: `ret_1`, `mid`
**Output**: `data/market/silver/volatility/risk_metrics.csv`

```python
def compute_volatility_features(df):
    """Generate volatility and risk metrics"""

    df = df.sort_values('time').copy()

    # Realized volatility (multiple windows)
    df['vol_5'] = df['ret_1'].rolling(5, min_periods=2).std() * np.sqrt(5)
    df['vol_20'] = df['ret_1'].rolling(20, min_periods=5).std() * np.sqrt(20)
    df['vol_60'] = df['ret_1'].rolling(60, min_periods=10).std() * np.sqrt(60)

    # Range-based volatility
    df['high_5'] = df['mid'].rolling(5, min_periods=2).max()
    df['low_5'] = df['mid'].rolling(5, min_periods=2).min()
    df['range_vol'] = (df['high_5'] - df['low_5']) / df['mid']

    # Volatility regime classification
    df['vol_percentile'] = df['vol_20'].rolling(100, min_periods=20).rank(pct=True)
    df['high_vol_regime'] = (df['vol_percentile'] > 0.8).astype(int)
    df['low_vol_regime'] = (df['vol_percentile'] < 0.2).astype(int)

    return df
```

**Features Created** (9 features):
- `vol_5`, `vol_20`, `vol_60`: Multi-window volatility
- `high_5`, `low_5`, `range_vol`: Range-based volatility
- `vol_percentile`: Volatility ranking
- `high_vol_regime`, `low_vol_regime`: Regime flags

#### **Target Label Creation**

```python
def create_target_label(df, horizon=5):
    """Create binary target for price direction prediction"""

    df = df.sort_values('time').copy()

    # Future price after horizon periods
    df['future_mid'] = df['mid'].shift(-horizon)

    # Binary classification: 1 if price goes up, 0 if down
    df['y'] = (df['future_mid'] > df['mid']).astype(int)

    # Remove last `horizon` rows (can't compute target)
    df.loc[df.index[-horizon:], 'y'] = np.nan

    return df
```

### **Load Phase**

**Destinations**: 3 separate CSV files (incremental append)

```python
def append_to_csv(df, output_path, exclude_cols=None):
    """Append new rows to CSV (handles headers)"""

    exclude_cols = exclude_cols or set()
    df_to_write = df[[col for col in df.columns if col not in exclude_cols]]

    # Write header only if file doesn't exist
    write_header = not output_path.exists()

    df_to_write.to_csv(
        output_path,
        mode='a',           # Append mode
        index=False,
        header=write_header
    )
```

**Output Files**:
1. `data/market/silver/technical_features/sgd_vs_majors.csv`
2. `data/market/silver/microstructure/depth_features.csv`
3. `data/market/silver/volatility/risk_metrics.csv`

**Output Schema Example** (technical_features.csv):
```
time,instrument,mid,spread,ret_1,ret_5,roll_vol_20,roll_mean_20,zscore_20,ewma_short,ewma_long,ewma_signal,spread_pct,spread_zscore,y
2025-01-15T08:00:00Z,USD_SGD,1.3452,0.0002,0.00015,0.00074,0.0023,1.3448,0.17,1.3451,1.3449,0.0002,0.000149,0.15,1
```

### **ETL Summary: Bronze → Silver**

| Aspect | Details |
|--------|---------|
| **Frequency** | Every 100 candles (flush interval) or on-demand |
| **Latency** | <2 seconds for 100 candles |
| **Input** | 1 NDJSON file (Bronze candles) |
| **Output** | 3 CSV files (technical, microstructure, volatility) |
| **Features Created** | 24 total (9 technical + 6 microstructure + 9 volatility) |
| **Transformations** | Feature engineering only (no filtering/imputation) |
| **Dependencies** | Per-instrument independent (horizontally scalable) |

---

## Market ETL 3: Silver → Gold

### **Extract Phase**

**Sources**: 3 Silver CSV files
**Implementation**: [`build_market_gold.py`](src/build_market_gold.py)

```python
def load_silver_features(technical_path, microstructure_path, volatility_path):
    """Load all Silver layer feature files"""

    # Load technical features
    technical_df = pd.read_csv(technical_path)
    technical_df['time'] = pd.to_datetime(technical_df['time'])

    # Load microstructure features
    microstructure_df = pd.read_csv(microstructure_path)
    microstructure_df['time'] = pd.to_datetime(microstructure_df['time'])

    # Load volatility features
    volatility_df = pd.read_csv(volatility_path)
    volatility_df['time'] = pd.to_datetime(volatility_df['time'])

    return technical_df, microstructure_df, volatility_df
```

### **Transform Phase**

**Transformations**: 4 major steps

#### **T1: Merge Silver Features**

**Purpose**: Combine 3 Silver CSVs into unified feature set

```python
def merge_market_features(technical_df, microstructure_df, volatility_df):
    """Merge all market features on (time, instrument)"""

    # Start with technical features as base (always complete)
    merged_df = technical_df.copy()

    # Left join microstructure (may have gaps)
    if not microstructure_df.empty:
        micro_cols = [col for col in microstructure_df.columns
                     if col not in ['time', 'instrument', 'y']]

        merged_df = merged_df.merge(
            microstructure_df[['time', 'instrument'] + micro_cols],
            on=['time', 'instrument'],
            how='left',  # Keep all market rows
            suffixes=('', '_micro')
        )

    # Left join volatility
    if not volatility_df.empty:
        vol_cols = [col for col in volatility_df.columns
                   if col not in ['time', 'instrument', 'y']]

        merged_df = merged_df.merge(
            volatility_df[['time', 'instrument'] + vol_cols],
            on=['time', 'instrument'],
            how='left',
            suffixes=('', '_vol')
        )

    return merged_df
```

**Result**: Single DataFrame with all Silver features

#### **T2: Add Cross-Instrument Features**

**Purpose**: Create features requiring multiple currency pairs

```python
def add_cross_instrument_features(df):
    """Add Gold-layer cross-instrument derived features"""

    instruments = df['instrument'].unique()

    if len(instruments) > 1:
        # Pivot returns for correlation calculation
        pivot_returns = df.pivot_table(
            index='time',
            columns='instrument',
            values='ret_1',
            fill_value=0
        )

        # USD_SGD vs EUR_USD correlation
        if 'USD_SGD' in pivot_returns.columns and 'EUR_USD' in pivot_returns.columns:
            rolling_corr = pivot_returns['USD_SGD'].rolling(50).corr(
                pivot_returns['EUR_USD']
            )

            corr_df = pd.DataFrame({
                'time': rolling_corr.index,
                'usd_sgd_eur_usd_corr': rolling_corr.values
            })

            df = df.merge(corr_df, on='time', how='left')

        # Relative performance vs major pairs basket
        if 'USD_SGD' in pivot_returns.columns:
            basket_return = pivot_returns.mean(axis=1)

            rel_perf = pd.DataFrame({
                'time': pivot_returns.index,
                'relative_performance': pivot_returns['USD_SGD'] - basket_return
            })

            df = df.merge(rel_perf, on='time', how='left')

    return df
```

**Features Created** (2 features):
- `usd_sgd_eur_usd_corr`: Rolling 50-period correlation
- `relative_performance`: USD_SGD return vs basket

#### **T3: Add Time Features**

**Purpose**: Trading session and temporal indicators

```python
def add_time_features(df):
    """Add time-based features"""

    # Extract time components
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Trading session indicators (UTC-based)
    # Note: SGT = UTC+8
    df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

    return df
```

**Features Created** (6 features):
- `hour`, `day_of_week`, `is_weekend`: Temporal features
- `asian_session`, `london_session`, `ny_session`: Session flags

#### **T4: Clean and Validate**

**Purpose**: Preprocessing for ML training readiness

```python
def clean_and_validate(df, min_obs_per_instrument=100):
    """Clean data and apply validation rules"""

    # 1. Remove instruments with insufficient data
    instrument_counts = df['instrument'].value_counts()
    valid_instruments = instrument_counts[instrument_counts >= min_obs_per_instrument].index
    df = df[df['instrument'].isin(valid_instruments)]

    # 2. Remove rows with missing target
    df = df.dropna(subset=['y'])

    # 3. Handle infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 4. Forward-fill missing microstructure features (from sparse order book)
    for col in numeric_cols:
        if col not in ['time', 'y']:
            # Forward fill within each instrument
            df[col] = df.groupby('instrument')[col].fillna(method='ffill')
            # Then fill remaining with median
            df[col] = df[col].fillna(df[col].median())

    # 5. Drop any remaining rows with NaN in features
    feature_cols = [col for col in df.columns if col not in ['time', 'instrument', 'y']]
    df = df.dropna(subset=feature_cols)

    return df
```

**Preprocessing Steps**:
1. ✅ Filter instruments (min 100 observations)
2. ✅ Remove rows without target
3. ✅ Replace infinities with NaN
4. ✅ Forward-fill sparse microstructure data
5. ✅ Median imputation for remaining NaNs
6. ✅ Drop rows with any remaining NaN

### **Load Phase**

**Destination**: `data/market/gold/training/market_features.csv`

```python
def save_gold_layer(df, output_path):
    """Save Gold layer to CSV"""

    # Sort by instrument and time
    df = df.sort_values(['instrument', 'time']).reset_index(drop=True)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
```

**Output Schema**:
```
time,instrument,mid,spread,ret_1,ret_5,roll_vol_20,roll_mean_20,zscore_20,
ewma_short,ewma_long,ewma_signal,spread_pct,spread_zscore,
total_liquidity,liquidity_imbalance,effective_spread,quoted_depth,avg_liquidity_20,liquidity_shock,
vol_5,vol_20,vol_60,high_5,low_5,range_vol,vol_percentile,high_vol_regime,low_vol_regime,
usd_sgd_eur_usd_corr,relative_performance,
hour,day_of_week,is_weekend,asian_session,london_session,ny_session,
y
```

### **ETL Summary: Silver → Gold**

| Aspect | Details |
|--------|---------|
| **Frequency** | On-demand or scheduled (e.g., hourly) |
| **Latency** | <5 seconds for 1000 rows |
| **Input** | 3 CSV files (Silver features) |
| **Output** | 1 CSV file (Gold training data) |
| **Features Total** | 32+ features (24 Silver + 8 Gold-derived) |
| **Transformations** | Merge, cross-instrument, time features, preprocessing |
| **Data Quality** | Missing value imputation, outlier handling, validation |

---

# News Data Medallion

## News ETL 1: Web Scraping → Bronze

### **Extract Phase**

**Sources**: Multiple financial news websites
**Implementation**: [`news_scraper.py`](src/news_scraper.py)

```python
class NewsSource:
    """Configuration for each news source"""
    def __init__(self, name, rss_url, base_url, selectors):
        self.name = name
        self.rss_url = rss_url
        self.base_url = base_url
        self.selectors = selectors  # CSS selectors for content extraction

# Configured sources
sources = [
    NewsSource(
        name="reuters_singapore",
        rss_url="https://www.reuters.com/markets/currencies/rss",
        base_url="https://www.reuters.com",
        selectors={
            "content": "div[data-module='ArticleBody'] p",
            "title": "h1",
            "timestamp": "time"
        }
    ),
    NewsSource(
        name="bloomberg_asia",
        rss_url="https://feeds.bloomberg.com/markets/news.rss",
        base_url="https://www.bloomberg.com",
        selectors={
            "content": ".body-content p",
            "title": "h1",
            "timestamp": "time"
        }
    ),
    NewsSource(
        name="channelnewsasia_business",
        rss_url="https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511",
        base_url="https://www.channelnewsasia.com",
        selectors={
            "content": ".text-long p",
            "title": "h1",
            "timestamp": "time"
        }
    ),
    NewsSource(
        name="straits_times_business",
        rss_url="https://www.straitstimes.com/rss-feeds",
        base_url="https://www.straitstimes.com",
        selectors={
            "content": ".story-content p",
            "title": "h1",
            "timestamp": "time"
        }
    )
]
```

**Extraction Process**:

```python
async def fetch_rss_feed(source):
    """Fetch RSS feed and parse entries"""

    async with aiohttp.ClientSession() as session:
        async with session.get(source.rss_url, timeout=30) as response:
            rss_content = await response.text()
            feed = feedparser.parse(rss_content)

            articles = []
            for entry in feed.entries:
                # Filter for recent articles (last 7 days)
                pub_time = datetime(*entry.published_parsed[:6])
                if datetime.now() - pub_time > timedelta(days=7):
                    continue

                articles.append({
                    'title': entry.title,
                    'url': entry.link,
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                    'source': source.name
                })

            return articles

async def scrape_article_content(url, source):
    """Scrape full article content from URL"""

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=30) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')

            # Extract content using source-specific CSS selectors
            content_elements = soup.select(source.selectors['content'])
            content = ' '.join([elem.get_text().strip() for elem in content_elements])

            return content
```

### **Transform Phase**

**Transformations**: Filtering and metadata enrichment

```python
def is_sgd_relevant(text):
    """Check if article is relevant to SGD"""

    # SGD-relevant keywords
    sgd_keywords = {
        # Singapore-specific
        'singapore', 'sgd', 'singapore dollar', 'monetary authority of singapore',
        'mas', 'usd/sgd', 'usd_sgd', 'singapore economy', 'singapore gdp',
        'singapore inflation', 'singapore trade', 'neer',

        # ASEAN / Regional
        'asean', 'southeast asia', 'asia pacific', 'asian markets',

        # Global FX drivers (affects USD side)
        'federal reserve', 'fed', 'fomc', 'dollar', 'usd',
        'interest rate', 'rate hike', 'rate cut', 'monetary policy',
        'central bank', 'inflation', 'cpi', 'employment',

        # FX market general
        'currency', 'forex', 'fx', 'exchange rate', 'safe haven'
    }

    text_lower = text.lower()
    return any(keyword in text_lower for keyword in sgd_keywords)

async def process_articles(source):
    """Process all articles from a source with relevance filtering"""

    # Fetch RSS feed
    rss_articles = await fetch_rss_feed(source)

    saved_count = 0
    for article in rss_articles:
        # Skip if already processed (check seen_articles set)
        if article['url'] in seen_articles:
            continue

        # Quick relevance check on title/summary
        quick_text = f"{article['title']} {article['summary']}"
        if not is_sgd_relevant(quick_text):
            continue

        # Scrape full content
        content = await scrape_article_content(article['url'], source)

        # Fallback to RSS summary if scraping blocked (paywall)
        if not content and article['summary']:
            content = article['summary']

        # Final relevance check on full content
        full_text = f"{article['title']} {article['summary']} {content}"
        if not is_sgd_relevant(full_text):
            continue

        # Transform to Bronze format
        bronze_article = {
            'url': article['url'],
            'title': article['title'],
            'content': content,
            'summary': article['summary'],
            'published': article['published'],
            'source': source.name,
            'scraped_at': datetime.now(timezone.utc).isoformat(),
            'sgd_relevant': True,
            'word_count': len(content.split()),
            'char_count': len(content)
        }

        yield bronze_article
```

**Transformations Applied**:
1. ✅ Relevance filtering (SGD keywords)
2. ✅ Deduplication (URL tracking)
3. ✅ Metadata enrichment (word_count, char_count)
4. ✅ Fallback handling (paywall → RSS summary)
5. ❌ No sentiment analysis (happens in Silver)

### **Load Phase**

**Destination**: `data/bronze/news/financial_news_2025.ndjson`
**Format**: NDJSON (append-only)

```python
def save_to_bronze(article, output_file):
    """Append article to NDJSON file"""

    with open(output_file, 'a') as f:
        f.write(json.dumps(article) + '\n')

    # Track seen URLs
    seen_articles.add(article['url'])
```

**Output Example**:
```json
{"url":"https://www.reuters.com/markets/currencies/mas-maintains-policy-stance-2025","title":"MAS maintains monetary policy stance amid inflation concerns","content":"The Monetary Authority of Singapore (MAS) announced today that it will maintain its current monetary policy stance...","summary":"Singapore's central bank maintains policy stance","published":"2025-01-15T07:30:00Z","source":"reuters_singapore","scraped_at":"2025-01-15T08:00:00.000000Z","sgd_relevant":true,"word_count":456,"char_count":2891}
```

### **Deduplication Strategy**

```python
def load_seen_articles(output_file):
    """Load URLs of already-processed articles"""
    seen = set()

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                article = json.loads(line)
                seen.add(article['url'])

    return seen
```

### **ETL Summary: Web Scraping → Bronze**

| Aspect | Details |
|--------|---------|
| **Frequency** | Every 30 minutes |
| **Latency** | ~1-2 seconds per article (scraping + parsing) |
| **Data Volume** | ~5-20 articles/day (after filtering) |
| **File Size** | ~3KB/article × 7,300/year = ~22 MB/year |
| **Transformations** | Relevance filtering + metadata enrichment |
| **Quality Checks** | Keyword matching, URL deduplication, content length validation |

---

## News ETL 2: Bronze → Silver (FinGPT)

### **Extract Phase**

**Source**: `data/bronze/news/financial_news_2025.ndjson`
**Implementation**: [`build_news_features.py`](src/build_news_features.py)

```python
def load_bronze_articles(input_dir):
    """Load raw articles from Bronze layer"""

    articles = []

    for file_path in Path(input_dir).glob('*.ndjson'):
        with open(file_path, 'r') as f:
            for line in f:
                article = json.loads(line.strip())
                articles.append({
                    'story_id': hashlib.md5(article['url'].encode()).hexdigest(),
                    'title': article['title'],
                    'content': article['content'],
                    'published_at': pd.to_datetime(article['published']),
                    'source': article['source'],
                    'url': article['url']
                })

    return pd.DataFrame(articles)
```

### **Transform Phase**

**Transformations**: FinGPT sentiment analysis with market context

#### **T1: Load Market Context**

```python
def get_latest_market_context(market_features_path, article_time):
    """Get market state at time of article publication"""

    # Load Market Silver features
    market_df = pd.read_csv(market_features_path)
    market_df['time'] = pd.to_datetime(market_df['time'])

    # Filter for USD_SGD
    usd_sgd_df = market_df[market_df['instrument'] == 'USD_SGD']

    # Find most recent market data before article time
    recent_market = usd_sgd_df[usd_sgd_df['time'] <= article_time].iloc[-1]

    # Extract context features
    market_context = {
        'mid': recent_market['mid'],
        'ret_5': recent_market['ret_5'],
        'vol_20': recent_market['vol_20'],
        'high_vol_regime': recent_market['high_vol_regime'],
        'spread_pct': recent_market['spread_pct'],
        'zscore_20': recent_market['zscore_20'],
        'session': get_trading_session(article_time.hour)
    }

    return market_context

def get_trading_session(hour):
    """Determine trading session from hour"""
    if 0 <= hour < 9:
        return 'asian'
    elif 9 <= hour < 17:
        return 'london'
    else:
        return 'ny'
```

#### **T2: FinGPT Analysis**

```python
def analyze_with_fingpt(article, market_context):
    """Run FinGPT sentiment analysis with market context"""

    # Initialize FinGPT processor
    processor = FinGPTProcessor(
        model_name="FinGPT/fingpt-sentiment_llama2-7b_lora",
        use_8bit=True  # Memory optimization
    )

    # Construct market-aware prompt
    prompt = f"""You are a financial analyst specializing in Singapore Dollar (SGD) trading.
Analyze the following news for SGD trading signals, considering BOTH the news content AND current market conditions.

Headline: {article['title']}
Article: {article['content']}

CURRENT MARKET STATE:
- USD/SGD Mid Price: {market_context['mid']:.4f}
- Recent 5-tick Return: {market_context['ret_5']:.2%}
- 20-period Volatility: {market_context['vol_20']:.2%}
- Volatility Regime: {'High' if market_context['high_vol_regime'] else 'Normal'}
- Spread (% of mid): {market_context['spread_pct']:.3%}
- Price Z-Score (20-period): {market_context['zscore_20']:.2f}
- Trading Session: {market_context['session']}

Consider these key questions:
1. How does this news relate to the current market state?
2. Has the market already priced in this information?
3. Does the news sentiment align with or diverge from current price action?
4. What is the expected impact given current volatility and market regime?

Provide analysis in this exact format:
SENTIMENT: [bullish/bearish/neutral]
CONFIDENCE: [0.0-1.0]
SGD_SIGNAL: [bullish/bearish/neutral]
POLICY: [hawkish/dovish/neutral]
TIMEFRAME: [immediate/short_term/medium_term]
MARKET_COHERENCE: [aligned/divergent/neutral]
ADJUSTED_STRENGTH: [0.0-1.0]
FACTORS: [key market drivers, separated by semicolons]

Analysis:"""

    # FinGPT inference
    response = processor.pipeline(
        prompt,
        max_length=1024,
        temperature=0.7,
        do_sample=True
    )

    # Parse response
    analysis = parse_fingpt_response(response[0]['generated_text'])

    return analysis

def parse_fingpt_response(raw_response):
    """Parse structured FinGPT output"""

    # Extract structured fields using regex
    sentiment_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}

    sentiment = extract_field(raw_response, 'SENTIMENT', sentiment_map)
    confidence = float(extract_field(raw_response, 'CONFIDENCE'))
    sgd_signal = extract_field(raw_response, 'SGD_SIGNAL', sentiment_map)
    policy = extract_field(raw_response, 'POLICY')
    timeframe = extract_field(raw_response, 'TIMEFRAME')
    coherence = extract_field(raw_response, 'MARKET_COHERENCE')
    adjusted_strength = float(extract_field(raw_response, 'ADJUSTED_STRENGTH'))
    factors = extract_field(raw_response, 'FACTORS').split(';')

    return {
        'sentiment_score': sentiment,
        'confidence': confidence,
        'sgd_directional_signal': sgd_signal,
        'policy_implications': policy,
        'time_horizon': timeframe,
        'market_coherence': coherence,
        'signal_strength_adjusted': adjusted_strength,
        'key_factors': factors,
        'raw_response': raw_response
    }
```

#### **T3: Entity and Topic Extraction**

```python
def extract_entities_and_topics(article):
    """Extract named entities and topic classifications"""

    # Simple keyword-based extraction (could use NER model)
    entities = {
        'currency_mentions': extract_currencies(article['content']),
        'central_bank_mentions': extract_central_banks(article['content']),
        'policy_keywords': extract_policy_terms(article['content'])
    }

    topics = {
        'topic_labels': classify_topics(article['content']),
        'volatility_hits': count_volatility_keywords(article['content'])
    }

    return entities, topics

def extract_currencies(text):
    """Extract currency mentions"""
    currency_patterns = r'\b(SGD|USD|EUR|GBP|JPY|CNY)\b'
    matches = re.findall(currency_patterns, text.upper())
    return ','.join(set(matches))

def classify_topics(text):
    """Simple topic classification"""
    topics = []

    if any(word in text.lower() for word in ['inflation', 'cpi', 'price']):
        topics.append('inflation')
    if any(word in text.lower() for word in ['rate', 'monetary policy', 'interest']):
        topics.append('monetary_policy')
    if any(word in text.lower() for word in ['trade', 'export', 'import']):
        topics.append('trade')
    if any(word in text.lower() for word in ['gdp', 'growth', 'recession']):
        topics.append('economic_growth')

    return ','.join(topics)
```

### **Load Phase**

**Destinations**: 3 separate CSV files

```python
def save_silver_features(sentiment_df, entity_df, topic_df, output_paths):
    """Save Silver layer features to separate CSVs"""

    # 1. Sentiment features (FinGPT outputs)
    sentiment_df.to_csv(output_paths['sentiment'], index=False)

    # 2. Entity features
    entity_df.to_csv(output_paths['entities'], index=False)

    # 3. Topic features
    topic_df.to_csv(output_paths['topics'], index=False)
```

**Output Files**:
1. `data/news/silver/sentiment_scores/sentiment_features.csv`
2. `data/news/silver/entity_mentions/entity_features.csv`
3. `data/news/silver/topic_signals/topic_features.csv`

**Output Schema** (sentiment_features.csv):
```
story_id,published_at,title,source,url,
sentiment_score,confidence,sgd_directional_signal,
policy_implications,time_horizon,market_coherence,signal_strength_adjusted,
market_mid_price,market_session,
key_factors
```

**Example Row**:
```
abc123,2025-01-15T07:30:00Z,"MAS maintains policy stance",reuters_singapore,https://...,
0.65,0.82,0.58,
hawkish,short_term,aligned,0.72,
1.3452,asian,
"stable inflation; strong sgd demand; regional stability"
```

### **ETL Summary: Bronze → Silver (FinGPT)**

| Aspect | Details |
|--------|---------|
| **Frequency** | Batch processing (every 6-12 hours) |
| **Latency** | ~2-5 seconds per article (FinGPT inference) |
| **Input** | 1 NDJSON file (Bronze articles) |
| **Output** | 3 CSV files (sentiment, entities, topics) |
| **Features Created** | 14+ features (10 FinGPT + 4 entity/topic) |
| **Transformations** | FinGPT analysis + NER + topic classification |
| **Dependencies** | Requires Market Silver for context (cross-pipeline dependency) |
| **GPU Requirements** | 8GB+ VRAM recommended (or fallback to lexicon-based) |

---

## News ETL 3: Silver → Gold

**(See earlier detailed description in the conversation)**

### **ETL Summary: Silver → Gold**

| Aspect | Details |
|--------|---------|
| **Frequency** | Hourly or on-demand |
| **Latency** | <3 seconds for 100 articles |
| **Input** | 3 CSV files (Silver sentiment, entities, topics) |
| **Output** | 1 CSV file (Gold trading signals) |
| **Transformations** | Merge, currency explosion, temporal aggregation, quality scoring |
| **Granularity Change** | Article-level → Hourly-level |
| **Features Created** | 24+ aggregated features |

---

# Combined Training Layer

## Combined ETL: Gold Layers → Combined Training

### **Extract Phase**

**Sources**: 2 Gold CSV files
**Implementation**: [`train_combined_model.py`](src/train_combined_model.py)

```python
def load_gold_data(market_path, news_path):
    """Load Gold layer data from both pipelines"""

    # Load market features
    market_df = pd.read_csv(market_path)
    market_df['time'] = pd.to_datetime(market_df['time'])

    # Load news signals
    news_df = pd.read_csv(news_path)
    news_df['signal_time'] = pd.to_datetime(news_df['signal_time'])

    return market_df, news_df
```

### **Transform Phase**

**Transformations**: As-of join + lag feature engineering

#### **T1: As-Of Join (Temporal Alignment)**

```python
def merge_market_news_features(market_df, news_df, focus_currency='USD_SGD', news_tolerance='6H'):
    """Merge market features with news signals using as-of join"""

    # Filter market data for focus currency
    market_currency = market_df[market_df['instrument'] == focus_currency].copy()

    # Filter news for relevant currency
    currency_code = focus_currency.split('_')[1]  # SGD from USD_SGD
    relevant_news = news_df[news_df['currency'] == currency_code].copy()

    # Sort both by time
    market_currency = market_currency.sort_values('time')
    relevant_news = relevant_news.sort_values('signal_time')

    # Convert tolerance to timedelta
    tolerance = pd.Timedelta(news_tolerance)  # e.g., "6H" → 6 hours

    # As-of join: For each market row, find most recent news within tolerance
    merged_rows = []

    for _, market_row in market_currency.iterrows():
        market_time = market_row['time']

        # Find most recent news within lookback window
        news_cutoff = market_time - tolerance
        eligible_news = relevant_news[
            (relevant_news['signal_time'] <= market_time) &
            (relevant_news['signal_time'] >= news_cutoff)
        ]

        if not eligible_news.empty:
            # Take most recent news
            latest_news = eligible_news.iloc[-1]

            # Merge row
            merged_row = market_row.to_dict()
            merged_row['news_sentiment'] = latest_news['avg_sentiment']
            merged_row['news_directional_signal'] = latest_news['avg_directional']
            merged_row['news_trading_signal'] = latest_news['trading_signal']
            merged_row['news_quality_score'] = latest_news['quality_score']
            merged_row['news_article_count'] = latest_news['article_count']
            merged_row['news_age_minutes'] = (market_time - latest_news['signal_time']).total_seconds() / 60
            merged_row['news_available'] = 1

        else:
            # No news available - use neutral defaults
            merged_row = market_row.to_dict()
            merged_row['news_sentiment'] = 0.0
            merged_row['news_directional_signal'] = 0.0
            merged_row['news_trading_signal'] = 0.0
            merged_row['news_quality_score'] = 0.0
            merged_row['news_article_count'] = 0
            merged_row['news_age_minutes'] = np.nan
            merged_row['news_available'] = 0

        merged_rows.append(merged_row)

    return pd.DataFrame(merged_rows)
```

**Key Insight**: As-of join ensures **no look-ahead bias**
- Market time: 2025-01-15 12:00
- News cutoff: 2025-01-15 06:00 (12:00 - 6H tolerance)
- Only news published **before** 12:00 and **after** 06:00 are considered

#### **T2: Lag Feature Engineering**

```python
def create_lag_features(df, lag_periods=[1, 2, 3, 5, 10]):
    """Create lagged features for time-series patterns"""

    df = df.sort_values('time').copy()

    # Market features to lag
    market_features = [
        'ret_1', 'ret_5', 'vol_20', 'spread_pct', 'zscore_20',
        'ewma_signal', 'liquidity_imbalance', 'high_vol_regime'
    ]

    # News features to lag
    news_features = [
        'news_sentiment', 'news_directional_signal', 'news_trading_signal',
        'news_quality_score'
    ]

    # Create lags
    for col in market_features + news_features:
        for lag in lag_periods:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    return df
```

**Example Output**:
```
# Original features
time, ret_1, news_sentiment, ...

# Lagged features (1, 2, 3, 5, 10 periods back)
ret_1_lag1, ret_1_lag2, ret_1_lag3, ret_1_lag5, ret_1_lag10,
news_sentiment_lag1, news_sentiment_lag2, ...
```

**Why XGBoost Needs Lag Features**:
- **LSTM**: Automatically learns temporal patterns from sequences
- **XGBoost**: Tree-based, no sequence memory → requires explicit lags
- **Benefit**: Explicit lags are interpretable (can see which lag matters most)

#### **T3: Final Preprocessing**

```python
def prepare_for_training(df):
    """Final preprocessing for XGBoost training"""

    # 1. Drop rows with NaN in lag features (first N rows)
    max_lag = 10
    df = df.iloc[max_lag:]

    # 2. Drop rows with missing target
    df = df.dropna(subset=['y'])

    # 3. Feature scaling (optional for XGBoost, but helps)
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col not in ['time', 'instrument', 'y']]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 4. Temporal train/test split (NO SHUFFLING)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    return train_df, test_df
```

### **Load Phase**

**Destination**: Temporary combined DataFrame (not saved to disk in most cases)

```python
# Combined features used directly for training
X_train = train_df[feature_cols]
y_train = train_df['y']

# Train XGBoost
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Save trained model
dump(model, 'data/combined/models/gradient_boosting_combined_model.pkl')
```

### **ETL Summary: Gold → Combined Training**

| Aspect | Details |
|--------|---------|
| **Frequency** | Daily or on-demand (model retraining) |
| **Latency** | <10 seconds for 10,000 rows |
| **Input** | 2 CSV files (Market Gold + News Gold) |
| **Output** | Trained model (PKL file) + metrics |
| **Transformations** | As-of join, lag engineering, scaling, temporal split |
| **Features Total** | 60+ (32 market + 10 news + 50 lagged features) |
| **Key Innovation** | As-of join prevents look-ahead bias, lag features enable XGBoost time-series learning |

---

## Complete Pipeline Flow Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                    MARKET DATA MEDALLION                          │
├──────────────────────────────────────────────────────────────────┤
│  OANDA API                                                        │
│    ↓ ETL 1 (Type conversion, spread calc)                        │
│  Bronze: usd_sgd_hourly_2025.ndjson (NDJSON, append-only)       │
│    ↓ ETL 2 (Feature engineering, 3 parallel streams)             │
│  Silver: technical.csv, microstructure.csv, volatility.csv       │
│    ↓ ETL 3 (Merge, cross-instrument, time features, cleaning)    │
│  Gold: market_features.csv (32 features, ML-ready)               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                    NEWS DATA MEDALLION                            │
├──────────────────────────────────────────────────────────────────┤
│  Web Scraping (Reuters, Bloomberg, CNA, ST)                       │
│    ↓ ETL 1 (Relevance filtering, deduplication)                  │
│  Bronze: financial_news_2025.ndjson (NDJSON, append-only)        │
│    ↓ ETL 2 (FinGPT analysis + Market Silver context)             │
│  Silver: sentiment.csv, entities.csv, topics.csv                 │
│    ↓ ETL 3 (Merge, currency explosion, temporal aggregation)     │
│  Gold: trading_signals.csv (24 features, hourly signals)         │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                    COMBINED TRAINING LAYER                        │
├──────────────────────────────────────────────────────────────────┤
│  Market Gold + News Gold                                          │
│    ↓ ETL 4 (As-of join, lag engineering, temporal split)         │
│  Combined Features (60+ features with lags)                       │
│    ↓ XGBoost Training                                             │
│  Trained Model: gradient_boosting_combined_model.pkl             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles Across All ETLs

| Principle | Implementation |
|-----------|----------------|
| **Immutable Bronze** | Never modify Bronze after collection, enables full reprocessing |
| **NDJSON for Streaming** | Append-only format for crash recovery and deduplication |
| **Separation of Concerns** | Market/News pipelines independent, 3 Silver CSVs for different dependencies |
| **Stateless Silver** | Per-instrument features, horizontally scalable |
| **Stateful Gold** | Cross-instrument and time-aware features |
| **No Look-Ahead Bias** | As-of join ensures only past news used for prediction |
| **Explicit Lag Features** | XGBoost time-series learning via manual lag engineering |
| **Quality Over Quantity** | Multiple validation checkpoints, missing value imputation |
| **Temporal Train/Test Split** | No shuffling, respects time order |

---

This document provides complete transparency into every transformation step across the entire dual medallion pipeline.
