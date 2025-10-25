from datetime import timedelta
from feast import Field, FeatureView, FileSource
from feast.types import Float32, Int64, String

# Offline source: we will generate Parquet alongside existing CSVs
market_gold_source = FileSource(
    name="market_gold_parquet",
    path="data/market/gold/training/market_features.parquet",
    timestamp_field="event_timestamp",
)

market_features_view = FeatureView(
    name="market_gold_features",
    entities=["instrument"],
    ttl=timedelta(days=7),
    schema=[
        Field(name="instrument", dtype=String),
        # Example feature columns (adjust to your actual columns)
        Field(name="ret_1h", dtype=Float32),
        Field(name="ret_4h", dtype=Float32),
        Field(name="vol_1d", dtype=Float32),
        Field(name="rsi_14", dtype=Float32),
        Field(name="orderbook_imbalance", dtype=Float32),
    ],
    online=True,
    source=market_gold_source,
)

