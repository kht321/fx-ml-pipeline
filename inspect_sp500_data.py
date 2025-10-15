#!/usr/bin/env python3
"""Inspect and analyze the downloaded S&P 500 data."""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def load_data(file_path):
    """Load NDJSON data into a pandas DataFrame."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def analyze_data(df):
    """Perform comprehensive analysis on the data."""
    print("=" * 80)
    print("S&P 500 HISTORICAL DATA ANALYSIS")
    print("=" * 80)
    print()

    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    # Basic statistics
    print("📊 BASIC STATISTICS")
    print("-" * 80)
    print(f"Total candles:        {len(df):,}")
    print(f"Date range:           {df['time'].min()} to {df['time'].max()}")
    print(f"Duration:             {(df['time'].max() - df['time'].min()).days} days")
    print(f"Instrument:           {df['instrument'].iloc[0]}")
    print(f"Granularity:          {df['granularity'].iloc[0]}")
    print()

    # Price statistics
    print("💰 PRICE STATISTICS")
    print("-" * 80)
    print(f"Price range:          ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Current close:        ${df['close'].iloc[-1]:.2f}")
    print(f"First close:          ${df['close'].iloc[0]:.2f}")
    print(f"Return:               {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Average close:        ${df['close'].mean():.2f}")
    print(f"Std deviation:        ${df['close'].std():.2f}")
    print()

    # Volume statistics
    print("📈 VOLUME STATISTICS")
    print("-" * 80)
    print(f"Total volume:         {df['volume'].sum():,}")
    print(f"Average volume:       {df['volume'].mean():.2f}")
    print(f"Max volume:           {df['volume'].max():,}")
    print(f"Min volume:           {df['volume'].min():,}")
    print()

    # Calculate returns
    df['return'] = df['close'].pct_change()
    df['abs_return'] = df['return'].abs()

    print("📉 VOLATILITY & RETURNS")
    print("-" * 80)
    print(f"Average return:       {df['return'].mean() * 100:.4f}%")
    print(f"Return std dev:       {df['return'].std() * 100:.4f}%")
    print(f"Max positive return:  {df['return'].max() * 100:.4f}%")
    print(f"Max negative return:  {df['return'].min() * 100:.4f}%")
    print(f"Sharpe ratio (1min):  {(df['return'].mean() / df['return'].std()):.4f}")
    print()

    # Time-based analysis
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['date'] = df['time'].dt.date

    print("🕐 TEMPORAL ANALYSIS")
    print("-" * 80)
    print(f"Unique trading days:  {df['date'].nunique():,}")
    print(f"Avg candles per day:  {len(df) / df['date'].nunique():.0f}")
    print()

    # Most active hours
    print("Most active hours (by volume):")
    hourly_volume = df.groupby('hour')['volume'].sum().sort_values(ascending=False).head(5)
    for hour, vol in hourly_volume.items():
        print(f"  {hour:02d}:00 - {vol:,} volume")
    print()

    # Data quality
    print("✅ DATA QUALITY")
    print("-" * 80)
    print(f"Missing values:       {df.isnull().sum().sum()}")
    print(f"Duplicate times:      {df['time'].duplicated().sum()}")
    print(f"Zero volume candles:  {(df['volume'] == 0).sum()} ({(df['volume'] == 0).sum() / len(df) * 100:.2f}%)")
    print()

    # Sample data
    print("📋 SAMPLE DATA (First 5 candles)")
    print("-" * 80)
    print(df[['time', 'open', 'high', 'low', 'close', 'volume']].head().to_string(index=False))
    print()

    print("📋 SAMPLE DATA (Last 5 candles)")
    print("-" * 80)
    print(df[['time', 'open', 'high', 'low', 'close', 'volume']].tail().to_string(index=False))
    print()

    # Daily statistics
    daily_stats = df.groupby('date').agg({
        'close': ['first', 'last', 'min', 'max'],
        'volume': 'sum'
    }).reset_index()
    daily_stats.columns = ['date', 'open', 'close', 'low', 'high', 'volume']
    daily_stats['return'] = daily_stats['close'].pct_change()

    print("📅 DAILY SUMMARY (Last 10 trading days)")
    print("-" * 80)
    print(daily_stats.tail(10).to_string(index=False))
    print()

    # Biggest moves
    print("🚀 TOP 10 BIGGEST MOVES (1-minute)")
    print("-" * 80)
    biggest_moves = df.nlargest(10, 'abs_return')[['time', 'open', 'close', 'return']]
    biggest_moves['return_%'] = biggest_moves['return'] * 100
    print(biggest_moves[['time', 'open', 'close', 'return_%']].to_string(index=False))
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return df

def main():
    data_file = Path("data/bronze/prices/spx500_usd_m1_historical.ndjson")

    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Please run the download script first.")
        return

    print(f"Loading data from {data_file}...")
    print()

    df = load_data(data_file)
    analyze_data(df)

if __name__ == "__main__":
    main()
