"""
Feast Feature Definitions for S&P 500 ML Pipeline

This defines features from the Gold layer for online serving.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, String, Int64

# Entity: Trading instrument (e.g., SPX500_USD)
instrument_entity = Entity(
    name="instrument",
    join_keys=["instrument"],
    value_type=ValueType.STRING,
    description="Trading instrument identifier"
)

# Data source: Market features from Gold layer
market_features_source = FileSource(
    path="../data_clean/gold/market/features/spx500_features.parquet",
    timestamp_field="time"
)

# Data source: News signals from Gold layer
news_signals_source = FileSource(
    path="../data_clean/gold/news/signals/sp500_trading_signals.parquet",
    timestamp_field="signal_time"
)

# Feature View: Market technical indicators
market_features_view = FeatureView(
    name="market_features",
    entities=[instrument_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="close", dtype=Float64),
        Field(name="rsi_14", dtype=Float64),
        Field(name="macd", dtype=Float64),
        Field(name="macd_signal", dtype=Float64),
        Field(name="bb_upper", dtype=Float64),
        Field(name="bb_middle", dtype=Float64),
        Field(name="bb_lower", dtype=Float64),
        Field(name="sma_7", dtype=Float64),
        Field(name="sma_14", dtype=Float64),
        Field(name="ema_7", dtype=Float64),
        Field(name="ema_14", dtype=Float64),
        Field(name="atr_14", dtype=Float64),
        Field(name="adx_14", dtype=Float64),
        Field(name="volatility_20", dtype=Float64),
    ],
    source=market_features_source,
    online=True,
)

# Feature View: News sentiment signals
news_signals_view = FeatureView(
    name="news_signals",
    entities=[instrument_entity],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="avg_sentiment", dtype=Float64),
        Field(name="signal_strength", dtype=Float64),
        Field(name="article_count", dtype=Int64),
    ],
    source=news_signals_source,
    online=True,
)
