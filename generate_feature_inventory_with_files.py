#!/usr/bin/env python3
"""Generate complete feature inventory with file locations for ML training."""

import csv
from pathlib import Path

# Define the feature inventory with file locations
features = [
    # Market Pipeline - Bronze Layer
    ("Market", "Bronze", "time", "datetime", "temporal", "Raw OANDA data", "Timestamp of the candle in UTC", "2020-10-13T15:25:00.000000000Z", "Direct from API", "data/bronze/prices/spx500_usd_m1_5years.ndjson", "NDJSON", "1,705,276", "✅ Available"),
    ("Market", "Bronze", "instrument", "string", "identifier", "Raw OANDA data", "Trading instrument identifier", "SPX500_USD", "Direct from API", "data/bronze/prices/spx500_usd_m1_5years.ndjson", "NDJSON", "1,705,276", "✅ Available"),
    ("Market", "Bronze", "granularity", "string", "identifier", "Raw OANDA data", "Time interval of the candle", "M1", "Direct from API", "data/bronze/prices/spx500_usd_m1_5years.ndjson", "NDJSON", "1,705,276", "✅ Available"),
    ("Market", "Bronze", "open", "float", "price", "Raw OANDA data", "Opening price of the candle", "3527.4", "Direct from API", "data/bronze/prices/spx500_usd_m1_5years.ndjson", "NDJSON", "1,705,276", "✅ Available"),
    ("Market", "Bronze", "high", "float", "price", "Raw OANDA data", "Highest price during candle", "3527.8", "Direct from API", "data/bronze/prices/spx500_usd_m1_5years.ndjson", "NDJSON", "1,705,276", "✅ Available"),
    ("Market", "Bronze", "low", "float", "price", "Raw OANDA data", "Lowest price during candle", "3526.4", "Direct from API", "data/bronze/prices/spx500_usd_m1_5years.ndjson", "NDJSON", "1,705,276", "✅ Available"),
    ("Market", "Bronze", "close", "float", "price", "Raw OANDA data", "Closing price of the candle", "3526.6", "Direct from API", "data/bronze/prices/spx500_usd_m1_5years.ndjson", "NDJSON", "1,705,276", "✅ Available"),
    ("Market", "Bronze", "volume", "integer", "volume", "Raw OANDA data", "Number of trades during candle", "109", "Direct from API", "data/bronze/prices/spx500_usd_m1_5years.ndjson", "NDJSON", "1,705,276", "✅ Available"),
    ("Market", "Bronze", "collected_at", "datetime", "metadata", "Raw OANDA data", "Download timestamp", "2025-10-12T15:25:15Z", "Direct from API", "data/bronze/prices/spx500_usd_m1_5years.ndjson", "NDJSON", "1,705,276", "✅ Available"),

    # Market Pipeline - Silver Layer
    ("Market", "Silver", "All Bronze features", "mixed", "all", "Preserved from Bronze", "All 9 Bronze features preserved", "various", "From Bronze", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "mid", "float", "price", "Derived from close", "Mid price", "3526.6", "close", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "best_bid", "float", "price", "Derived from close", "Best bid (approximation)", "3526.6", "close", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "best_ask", "float", "price", "Derived from close", "Best ask (approximation)", "3526.6", "close", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "ret_1", "float", "returns", "Pct change", "1-period return", "0.00012", "(mid[t]-mid[t-1])/mid[t-1]", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "ret_5", "float", "returns", "Pct change", "5-period return", "0.00056", "(mid[t]-mid[t-5])/mid[t-5]", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "ret_10", "float", "returns", "Pct change", "10-period return", "0.00089", "(mid[t]-mid[t-10])/mid[t-10]", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "roll_vol_20", "float", "volatility", "Rolling std", "Rolling volatility (20)", "0.00026", "ret_1.rolling(20).std()", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "roll_vol_50", "float", "volatility", "Rolling std", "Rolling volatility (50)", "0.00027", "ret_1.rolling(50).std()", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "roll_mean_20", "float", "moving_average", "Rolling mean", "Simple MA (20)", "3528.5", "mid.rolling(20).mean()", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "zscore_20", "float", "technical", "Z-score", "Standardized price", "0.45", "(mid-roll_mean_20)/roll_vol_20", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "ewma_short", "float", "moving_average", "EWMA", "EWMA short (12)", "3527.2", "mid.ewm(span=12).mean()", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "ewma_long", "float", "moving_average", "EWMA", "EWMA long (26)", "3526.8", "mid.ewm(span=26).mean()", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "ewma_signal", "float", "technical", "MACD signal", "EWMA short - EWMA long", "0.4", "ewma_short - ewma_long", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "hl_range", "float", "price", "H-L difference", "High-low range", "1.4", "high - low", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "hl_range_pct", "float", "returns", "H-L percentage", "H-L as % of close", "0.00039", "(high-low)/close", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "high_vol_regime", "boolean", "volatility", "Vol threshold", "High vol indicator", "True", "roll_vol_20>threshold", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "momentum_5", "float", "technical", "Price momentum", "5-period momentum", "2.3", "mid - mid.shift(5)", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "momentum_20", "float", "technical", "Price momentum", "20-period momentum", "5.8", "mid - mid.shift(20)", "data/sp500/silver/technical_features/sp500_technical.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "volume_ma_20", "float", "volume", "Volume MA", "20-period vol MA", "95.5", "volume.rolling(20).mean()", "data/sp500/silver/microstructure/sp500_microstructure.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "volume_ratio", "float", "volume", "Volume ratio", "Vol / MA ratio", "1.14", "volume/volume_ma_20", "data/sp500/silver/microstructure/sp500_microstructure.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "volume_zscore", "float", "volume", "Volume z-score", "Standardized volume", "-0.23", "(vol-vol_ma)/vol_std", "data/sp500/silver/microstructure/sp500_microstructure.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "cc_vol_20", "float", "volatility", "Close-close vol", "CC volatility (20)", "0.00026", "Same as roll_vol_20", "data/sp500/silver/volatility/sp500_volatility.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "cc_vol_50", "float", "volatility", "Close-close vol", "CC volatility (50)", "0.00027", "Same as roll_vol_50", "data/sp500/silver/volatility/sp500_volatility.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "parkinson_vol", "float", "volatility", "Parkinson estimator", "H-L volatility", "0.00018", "sqrt(1/(4ln2)*(ln(H/L))^2)", "data/sp500/silver/volatility/sp500_volatility.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "gk_vol", "float", "volatility", "Garman-Klass est", "OHLC volatility", "0.00017", "GK formula on OHLC", "data/sp500/silver/volatility/sp500_volatility.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "realized_vol", "float", "volatility", "Realized vol", "From H-L range", "0.00039", "hl_range_pct", "data/sp500/silver/volatility/sp500_volatility.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "realized_vol_ma", "float", "volatility", "Realized vol MA", "MA of realized vol", "0.00029", "realized_vol.rolling(20).mean()", "data/sp500/silver/volatility/sp500_volatility.csv", "CSV", "1,705,276", "✅ Available"),
    ("Market", "Silver", "vol_of_vol", "float", "volatility", "Vol of vol", "2nd order vol", "0.00003", "roll_vol_20.rolling(50).std()", "data/sp500/silver/volatility/sp500_volatility.csv", "CSV", "1,705,276", "✅ Available"),

    # Market Pipeline - Gold Layer
    ("Market", "Gold", "All 37 features", "mixed", "all", "Merged Silver layers", "All Silver features consolidated", "various", "From Silver (3 files merged)", "data/sp500/gold/training/sp500_features.csv", "CSV", "1,705,276", "✅ Available for Training"),
    ("Market", "Gold", "All 37 features", "mixed", "all", "Exported for analysis", "Full feature set", "various", "From Gold layer", "outputs/feature_csvs/sp500_all_features.csv", "CSV", "1,705,276", "✅ Available for Training"),

    # News Pipeline - Bronze Layer
    ("News", "Bronze", "article_id", "string", "identifier", "Web scraping", "Article ID", "news_20201013_1234", "Generated", "data/news/bronze/raw_articles/*.json", "JSON", "N/A", "⚪ Not Collected"),
    ("News", "Bronze", "published_at", "datetime", "temporal", "From source", "Publication time", "2020-10-13T14:30:00Z", "From API/scrape", "data/news/bronze/raw_articles/*.json", "JSON", "N/A", "⚪ Not Collected"),
    ("News", "Bronze", "source", "string", "metadata", "From source", "News publisher", "Bloomberg", "From API/scrape", "data/news/bronze/raw_articles/*.json", "JSON", "N/A", "⚪ Not Collected"),
    ("News", "Bronze", "headline", "string", "text", "From source", "Article title", "S&P 500 Reaches...", "From API/scrape", "data/news/bronze/raw_articles/*.json", "JSON", "N/A", "⚪ Not Collected"),
    ("News", "Bronze", "content", "string", "text", "From source", "Full article text", "The S&P 500...", "From API/scrape", "data/news/bronze/raw_articles/*.json", "JSON", "N/A", "⚪ Not Collected"),

    # News Pipeline - Silver Layer
    ("News", "Silver", "sentiment_score", "float", "sentiment", "FinGPT analysis", "Combined sentiment", "0.71", "FinGPT(text)->[-1,1]", "data/news/silver/sentiment_scores/sentiment_features.csv", "CSV", "N/A", "⚪ Awaiting News Data"),
    ("News", "Silver", "entities_mentioned", "array", "entities", "NER extraction", "Named entities", "[Apple, Fed]", "NER(content)", "data/news/silver/entity_mentions/entity_features.csv", "CSV", "N/A", "⚪ Awaiting News Data"),
    ("News", "Silver", "topic_general", "string", "topics", "Topic modeling", "Main topic", "earnings", "LDA/classification", "data/news/silver/topic_signals/topic_features.csv", "CSV", "N/A", "⚪ Awaiting News Data"),

    # News Pipeline - Gold Layer
    ("News", "Gold", "sentiment_aggregated", "float", "sentiment", "Time-bucketed agg", "Bucketed sentiment", "0.68", "Mean in time window", "data/news/gold/news_signals/trading_signals.csv", "CSV", "N/A", "⚪ Awaiting News Data"),
    ("News", "Gold", "signal_strength", "float", "signals", "Combined signal", "Trading signal", "0.68", "Weighted combination", "data/news/gold/news_signals/trading_signals.csv", "CSV", "N/A", "⚪ Awaiting News Data"),
    ("News", "Gold", "news_volume", "integer", "volume", "Article count", "News volume", "12", "Count in window", "data/news/gold/news_signals/trading_signals.csv", "CSV", "N/A", "⚪ Awaiting News Data"),
]

# Write to CSV
output_file = Path("outputs/COMPLETE_FEATURE_INVENTORY_WITH_FILES.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Write header
    writer.writerow([
        "pipeline", "layer", "feature_name", "data_type", "category",
        "source_transformation", "description", "example_value", "formula",
        "file_location", "file_format", "sample_count", "status"
    ])

    # Write data
    for row in features:
        writer.writerow(row)

print(f"✓ Feature inventory created: {output_file}")
print(f"  Total features documented: {len(features)}")
print(f"\nFeatures by status:")
print(f"  ✅ Available: {sum(1 for f in features if '✅' in f[12])}")
print(f"  ⚪ Not Yet Available: {sum(1 for f in features if '⚪' in f[12])}")
