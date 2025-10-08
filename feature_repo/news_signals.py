from datetime import timedelta
from feast import Field, FeatureView, FileSource
from feast.types import Float32, String

news_signals_source = FileSource(
    name="news_gold_parquet",
    path="data/news/gold/news_signals/trading_signals.parquet",
    timestamp_field="event_timestamp",
)

news_signals_view = FeatureView(
    name="news_gold_signals",
    entities=["instrument"],
    ttl=timedelta(days=3),
    schema=[
        Field(name="instrument", dtype=String),
        # Example signal columns (adjust to your actual columns)
        Field(name="news_sentiment_score", dtype=Float32),
        Field(name="news_signal_strength", dtype=Float32),
    ],
    online=True,
    source=news_signals_source,
)

