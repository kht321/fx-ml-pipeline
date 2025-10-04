"""Transform news Silver features into Gold-layer signals using FinGPT analysis.

This script consolidates Silver-layer news features (sentiment, entities, topics)
into unified Gold-layer trading signals optimized for market prediction models.
The Gold layer emphasizes actionable signals over raw sentiment.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Build the argument parser for news Gold layer processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentiment-features",
        type=Path,
        default=Path("data/news/silver/sentiment_scores/sentiment_features.csv"),
        help="CSV containing sentiment analysis results",
    )
    parser.add_argument(
        "--entity-features",
        type=Path,
        default=Path("data/news/silver/entity_mentions/entity_features.csv"),
        help="CSV containing entity and mention features",
    )
    parser.add_argument(
        "--topic-features",
        type=Path,
        default=Path("data/news/silver/topic_signals/topic_features.csv"),
        help="CSV containing topic signal features",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/news/gold/news_signals/trading_signals.csv"),
        help="Destination CSV for Gold news trading signals",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Hours of news history to aggregate for signals",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for including signals",
    )
    parser.add_argument(
        "--focus-currencies",
        nargs="*",
        default=["sgd", "usd", "eur", "gbp"],
        help="Currency codes to focus signal generation on",
    )
    return parser.parse_args(list(argv))


def load_news_features(sentiment_path: Path,
                      entity_path: Path,
                      topic_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all Silver layer news feature files."""

    dfs = {}

    # Load sentiment features
    if sentiment_path.exists():
        sentiment_df = pd.read_csv(sentiment_path)
        sentiment_df['published_at'] = pd.to_datetime(sentiment_df['published_at'])
        dfs['sentiment'] = sentiment_df
    else:
        dfs['sentiment'] = pd.DataFrame()

    # Load entity features
    if entity_path.exists():
        entity_df = pd.read_csv(entity_path)
        entity_df['published_at'] = pd.to_datetime(entity_df['published_at'])
        dfs['entity'] = entity_df
    else:
        dfs['entity'] = pd.DataFrame()

    # Load topic features
    if topic_path.exists():
        topic_df = pd.read_csv(topic_path)
        topic_df['published_at'] = pd.to_datetime(topic_df['published_at'])
        dfs['topic'] = topic_df
    else:
        dfs['topic'] = pd.DataFrame()

    return dfs['sentiment'], dfs['entity'], dfs['topic']


def merge_news_features(sentiment_df: pd.DataFrame,
                       entity_df: pd.DataFrame,
                       topic_df: pd.DataFrame) -> pd.DataFrame:
    """Merge all news features on story_id."""

    if sentiment_df.empty:
        return pd.DataFrame()

    # Start with sentiment as base
    merged_df = sentiment_df.copy()

    # Merge entity features
    if not entity_df.empty:
        entity_cols = [col for col in entity_df.columns
                      if col not in ['published_at']]
        merged_df = merged_df.merge(
            entity_df[entity_cols],
            on='story_id',
            how='left',
            suffixes=('', '_entity')
        )

    # Merge topic features
    if not topic_df.empty:
        topic_cols = [col for col in topic_df.columns
                     if col not in ['published_at', 'story_id']]
        topic_merge = topic_df[['story_id'] + topic_cols]
        merged_df = merged_df.merge(
            topic_merge,
            on='story_id',
            how='left',
            suffixes=('', '_topic')
        )

    return merged_df


def create_currency_signals(df: pd.DataFrame, focus_currencies: List[str]) -> pd.DataFrame:
    """Create currency-specific trading signals from news."""

    if df.empty:
        return pd.DataFrame()

    signals = []

    for _, row in df.iterrows():
        # Parse currency mentions
        currency_mentions = set()
        if pd.notna(row.get('currency_mentions', '')):
            currency_mentions = set(row['currency_mentions'].lower().split(','))

        # Determine affected currencies
        affected_currencies = currency_mentions.intersection(set(focus_currencies))

        # If SGD-specific news, always include SGD
        if row.get('mentions_sgd', False):
            affected_currencies.add('sgd')

        # If no specific currency mentioned, treat as broad market signal
        if not affected_currencies:
            affected_currencies = set(focus_currencies)

        # Create signal for each affected currency
        for currency in affected_currencies:
            signal = {
                'story_id': row['story_id'],
                'published_at': row['published_at'],
                'currency': currency.upper(),
                'headline': row.get('headline', ''),
                'source': row.get('source', ''),
            }

            # Primary sentiment signal (prefer FinGPT if available)
            signal['sentiment_score'] = row.get('sentiment_score', 0.0)
            signal['confidence'] = row.get('confidence', 0.0)

            # FinGPT-specific signals
            if 'sgd_directional_signal' in row and currency.lower() == 'sgd':
                signal['directional_signal'] = row['sgd_directional_signal']
            else:
                signal['directional_signal'] = signal['sentiment_score']

            # Policy implications
            signal['policy_tone'] = row.get('policy_implications', row.get('policy_tone', 'neutral'))
            signal['time_horizon'] = row.get('time_horizon', 'unknown')

            # Relevance scoring
            relevance = 1.0 if currency.lower() == 'sgd' and row.get('mentions_sgd', False) else 0.7
            if currency.lower() in currency_mentions:
                relevance = max(relevance, 0.8)

            signal['relevance_score'] = relevance
            signal['weighted_sentiment'] = signal['sentiment_score'] * relevance * signal['confidence']

            # Volatility and risk signals
            signal['volatility_signal'] = row.get('volatility_hits', 0) > 0
            signal['high_impact'] = (
                signal['confidence'] > 0.7 and
                abs(signal['sentiment_score']) > 0.5 and
                relevance > 0.8
            )

            signals.append(signal)

    return pd.DataFrame(signals)


def aggregate_temporal_signals(df: pd.DataFrame, lookback_hours: int = 24) -> pd.DataFrame:
    """Aggregate news signals over time windows for each currency."""

    if df.empty:
        return pd.DataFrame()

    # Sort by time
    df = df.sort_values('published_at').copy()

    # Create time windows
    max_time = df['published_at'].max()
    time_windows = pd.date_range(
        start=max_time - pd.Timedelta(hours=lookback_hours * 10),  # More history for rolling
        end=max_time,
        freq='1H'
    )

    aggregated_signals = []

    for currency in df['currency'].unique():
        currency_df = df[df['currency'] == currency].copy()

        for window_end in time_windows:
            window_start = window_end - pd.Timedelta(hours=lookback_hours)

            # Get articles in this window
            window_articles = currency_df[
                (currency_df['published_at'] >= window_start) &
                (currency_df['published_at'] <= window_end)
            ]

            if len(window_articles) == 0:
                continue

            # Aggregate signals
            agg_signal = {
                'signal_time': window_end,
                'currency': currency,
                'lookback_hours': lookback_hours,
            }

            # Weighted aggregations
            total_weight = window_articles['confidence'].sum()
            if total_weight > 0:
                agg_signal['avg_sentiment'] = (
                    window_articles['weighted_sentiment'].sum() / total_weight
                )
                agg_signal['avg_directional'] = (
                    (window_articles['directional_signal'] * window_articles['confidence']).sum() / total_weight
                )
            else:
                agg_signal['avg_sentiment'] = 0.0
                agg_signal['avg_directional'] = 0.0

            # Count-based features
            agg_signal['article_count'] = len(window_articles)
            agg_signal['high_confidence_count'] = len(window_articles[window_articles['confidence'] > 0.7])
            agg_signal['high_impact_count'] = window_articles['high_impact'].sum()
            agg_signal['volatility_mentions'] = window_articles['volatility_signal'].sum()

            # Policy tone consensus
            policy_modes = window_articles['policy_tone'].value_counts()
            agg_signal['dominant_policy_tone'] = policy_modes.index[0] if len(policy_modes) > 0 else 'neutral'
            agg_signal['policy_consensus'] = policy_modes.iloc[0] / len(window_articles) if len(policy_modes) > 0 else 0.0

            # Time horizon analysis
            time_horizons = window_articles['time_horizon'].value_counts()
            agg_signal['dominant_time_horizon'] = time_horizons.index[0] if len(time_horizons) > 0 else 'unknown'

            # Signal strength and direction
            strong_bullish = len(window_articles[window_articles['sentiment_score'] > 0.5])
            strong_bearish = len(window_articles[window_articles['sentiment_score'] < -0.5])

            agg_signal['signal_strength'] = abs(agg_signal['avg_sentiment'])
            agg_signal['signal_direction'] = np.sign(agg_signal['avg_sentiment'])
            agg_signal['signal_consensus'] = max(strong_bullish, strong_bearish) / len(window_articles)

            # Recent vs older news weighting
            recent_cutoff = window_end - pd.Timedelta(hours=6)
            recent_articles = window_articles[window_articles['published_at'] >= recent_cutoff]

            if len(recent_articles) > 0:
                recent_weight = recent_articles['confidence'].sum()
                if recent_weight > 0:
                    agg_signal['recent_sentiment'] = (
                        recent_articles['weighted_sentiment'].sum() / recent_weight
                    )
                else:
                    agg_signal['recent_sentiment'] = 0.0
                agg_signal['recent_article_count'] = len(recent_articles)
            else:
                agg_signal['recent_sentiment'] = agg_signal['avg_sentiment']
                agg_signal['recent_article_count'] = 0

            # Add metadata
            latest_story = window_articles.iloc[-1]
            agg_signal['latest_headline'] = latest_story['headline']
            agg_signal['latest_source'] = latest_story['source']
            agg_signal['minutes_since_latest'] = (window_end - latest_story['published_at']).total_seconds() / 60

            aggregated_signals.append(agg_signal)

    return pd.DataFrame(aggregated_signals)


def create_trading_signals(df: pd.DataFrame, min_confidence: float = 0.3) -> pd.DataFrame:
    """Create final trading signals with quality filters."""

    if df.empty:
        return df

    df = df.copy()

    # Quality score combining multiple factors
    df['quality_score'] = (
        df['signal_strength'] * 0.4 +
        df['signal_consensus'] * 0.3 +
        df['policy_consensus'] * 0.2 +
        np.minimum(df['article_count'] / 10, 1.0) * 0.1
    )

    # Trading signal strength
    df['trading_signal'] = np.where(
        df['quality_score'] >= min_confidence,
        df['signal_direction'] * df['signal_strength'],
        0.0
    )

    # Signal categories
    df['signal_category'] = pd.cut(
        abs(df['trading_signal']),
        bins=[0, 0.2, 0.5, 1.0],
        labels=['weak', 'moderate', 'strong'],
        include_lowest=True
    )

    # Time decay factor (recent news weighted higher)
    max_age_hours = 24
    df['age_hours'] = df['minutes_since_latest'] / 60
    df['time_decay'] = np.exp(-df['age_hours'] / max_age_hours)
    df['decayed_signal'] = df['trading_signal'] * df['time_decay']

    return df


def log(message: str) -> None:
    """Emit structured progress messages to stderr."""
    sys.stderr.write(f"[build_news_gold] {message}\n")
    sys.stderr.flush()


def main(argv: Iterable[str] | None = None) -> None:
    """Main processing function for news Gold layer."""
    args = parse_args(argv or sys.argv[1:])

    log("Loading Silver layer news features")

    # Load all Silver features
    sentiment_df, entity_df, topic_df = load_news_features(
        args.sentiment_features,
        args.entity_features,
        args.topic_features
    )

    if sentiment_df.empty:
        log("No sentiment features found - cannot proceed")
        sys.exit(1)

    log(f"Loaded: {len(sentiment_df)} sentiment, {len(entity_df)} entity, "
        f"{len(topic_df)} topic observations")

    # Merge all features
    log("Merging news features")
    merged_df = merge_news_features(sentiment_df, entity_df, topic_df)

    if merged_df.empty:
        log("No data after merging - cannot proceed")
        sys.exit(1)

    # Create currency-specific signals
    log("Creating currency-specific signals")
    currency_signals = create_currency_signals(merged_df, args.focus_currencies)

    if currency_signals.empty:
        log("No currency signals generated")
        sys.exit(1)

    # Filter by minimum confidence
    high_confidence = currency_signals[currency_signals['confidence'] >= args.min_confidence]
    log(f"Filtered to {len(high_confidence)} high-confidence signals from {len(currency_signals)} total")

    # Aggregate temporal signals
    log("Aggregating temporal signals")
    temporal_signals = aggregate_temporal_signals(high_confidence, args.lookback_hours)

    if temporal_signals.empty:
        log("No temporal signals generated")
        sys.exit(1)

    # Create final trading signals
    log("Creating trading signals")
    trading_signals = create_trading_signals(temporal_signals, args.min_confidence)

    # Sort by time and currency
    trading_signals = trading_signals.sort_values(['currency', 'signal_time']).reset_index(drop=True)

    # Save to Gold layer
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trading_signals.to_csv(args.output, index=False)

    # Summary statistics
    currencies = trading_signals['currency'].unique()
    signal_count = len(trading_signals)

    log(f"Gold layer complete: {signal_count} trading signals for {len(currencies)} currencies")

    for currency in currencies:
        curr_signals = trading_signals[trading_signals['currency'] == currency]
        strong_signals = len(curr_signals[curr_signals['signal_category'] == 'strong'])
        avg_strength = curr_signals['trading_signal'].abs().mean()

        log(f"  {currency}: {len(curr_signals)} signals, {strong_signals} strong, "
            f"avg strength: {avg_strength:.3f}")


if __name__ == "__main__":
    main()