from feast import Entity

# Entity definitions for Feast

# Instrument (e.g., currency pair: USD_SGD)
instrument = Entity(
    name="instrument",
    join_keys=["instrument"],
    description="Currency pair or instrument identifier",
)

