"""Utility functions for talking to the OANDA v20 REST API.

This module centralises API session initialisation and wraps the handful of
endpoints the project needs. Keeping all network calls here gives the other
modules a clean, dependency-free surface area and makes it easier to extend or
mock during tests.
"""

import os
from typing import Dict, Generator, Iterable, Tuple

from dotenv import load_dotenv
import oandapyV20
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.endpoints.instruments import InstrumentsCandles, InstrumentsOrderBook

# Populate process environment from a local ``.env`` file so that the SDK can
# locate the access token, account identifier, and environment (practice/live)
# without hard-coding secrets in the source tree.
load_dotenv()

API = oandapyV20.API(
    access_token=os.getenv("OANDA_TOKEN"),
    environment=os.getenv("OANDA_ENV", "practice"),
)


def stream_prices(account_id: str, instruments: Iterable[str]) -> Generator[Tuple[str, Dict[str, object]], None, None]:
    """Subscribe to the pricing stream and yield messages as they arrive.

    Parameters
    ----------
    account_id:
        OANDA account identifier. The stream is account-scoped so we need it on
        every request.
    instruments:
        Iterable of instrument symbols (``EUR_USD``, ``USD_SGD``, …) to stream.

    Yields
    ------
    tuple[str, dict]
        Message type (``"price"`` or ``"heartbeat"``) and the raw JSON payload
        returned by the SDK.
    """

    params = {"instruments": ",".join(instruments)}
    request = PricingStream(accountID=account_id, params=params)

    for message in API.request(request):
        # API.request yields dicts with a 'type' key like 'PRICE' or 'HEARTBEAT'
        msg_type = str(message.get("type", "")).lower()
        yield msg_type, message


def fetch_candles(
    instrument: str,
    granularity: str = "M1",
    count: int = 500,
    price: str = "MBA",
) -> dict:
    """Return historical OHLCV candles for the requested instrument.

    The SDK handles pagination and request signing, so this helper simply
    prepares the correct parameter dictionary and forwards the response body.
    """

    params = {"granularity": granularity, "count": count, "price": price}
    request = InstrumentsCandles(instrument=instrument, params=params)
    return API.request(request)


def fetch_orderbook(instrument: str) -> dict:
    """Retrieve the top-of-book snapshot for supported instruments."""

    request = InstrumentsOrderBook(instrument=instrument)
    return API.request(request)
