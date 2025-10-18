"""
Streamlit Protyping For FX-ML-Pipeline
"""
import streamlit as st
import os
import time
import pandas as pd, numpy as np
from oanda_api import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import plotly.graph_objects as go

OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID") 
OANDA_TOKEN = os.getenv("OANDA_TOKEN")

def get_latest_price(account_id, instrument="SPX500_USD"):
    params = {"instruments": instrument}
    request = pricing.PricingInfo(accountID=account_id, params=params)
    response = API.request(request)
    prices = response['prices'][0]
    bid = float(prices['bids'][0]['price'])
    ask = float(prices['asks'][0]['price'])
    return (bid + ask) / 2

def get_candles(instrument="SPX500_USD", granularity="H1", count=100):
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M"  # Midpoint prices
    }
    request = instruments.InstrumentsCandles(instrument=instrument, params=params)
    response = API.request(request)
    time.sleep(0.5)
    candles = response.get("candles", [])
    return candles

def main():
    st.set_page_config(page_title="S&P500 Volatility Dashboard", layout="wide")
    st.title("S&P500 Volatility Dashboard")

    # ---- SIDEBAR ----
    st.sidebar.header("Settings")
    instrument = st.sidebar.text_input("Instrument", "SPX500_USD")
    granularity = st.sidebar.selectbox("Granularity", ["M15", "H1", "D"], index=1)
    refresh = st.sidebar.slider("Auto-refresh (seconds)", 10, 300, 60)

    # ---- FETCH OANDA DATA ----
    st.header(f"OANDA Live Data: {instrument}")
    price = get_latest_price(OANDA_ACCOUNT_ID, instrument)
    st.metric("Current Price", f"{price:.2f} USD")

    candles = get_candles(instrument, granularity)
    df = pd.DataFrame([
        {
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
        } for c in candles
    ])

    # ---- CHART ----
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])
    st.plotly_chart(fig, use_container_width=True)

    # ---- ML INFERENCE ----
    # st.subheader("Predicted Volatility (Model Output)")
    # features = df[["open", "high", "low", "close"]].tail(1)
    # vol_pred = predict_volatility(features)
    # st.metric("Predicted Volatility", f"{vol_pred[0]:.4f}")

if __name__ == "__main__":
    main()