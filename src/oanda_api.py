"""Client helpers for interacting with the OANDA v20 REST API."""

import os
from typing import Generator, Iterable, Tuple

from dotenv import load_dotenv
import oandapyV20
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.endpoints.instruments import InstrumentsCandles, InstrumentsOrderBook

load_dotenv()

API = oandapyV20.API(
    access_token=os.getenv("OANDA_TOKEN"),
    environment=os.getenv("OANDA_ENV", "practice"),
)


def stream_prices(account_id: str, instruments: Iterable[str]) -> Generator[Tuple[str, dict], None, None]:
    """Yield streaming price messages for the requested instruments."""

    params = {"instruments": ",".join(instruments)}
    request = PricingStream(accountID=account_id, params=params)

    for msg_type, msg in API.request(request):
        yield msg_type, msg


def fetch_candles(instrument: str, granularity: str = "M1", count: int = 500) -> dict:
    """Fetch historical candles for a given instrument."""

    params = {"granularity": granularity, "count": count, "price": "MBA"}
    request = InstrumentsCandles(instrument=instrument, params=params)
    return API.request(request)


def fetch_orderbook(instrument: str) -> dict:
    """Fetch the current order book snapshot for an instrument."""

    request = InstrumentsOrderBook(instrument=instrument)
    return API.request(request)
