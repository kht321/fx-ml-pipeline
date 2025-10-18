#!/usr/bin/env python3
"""Add sentiment scores to existing news articles.

This script analyzes news articles and adds sentiment_score fields
based on the presence of positive/negative keywords in the headline and body.
"""

import json
import re
from pathlib import Path
from typing import Dict, List
import argparse


# Sentiment keyword dictionaries
POSITIVE_KEYWORDS = {
    # Strong positive
    'surge', 'surges', 'surging', 'soar', 'soars', 'soaring', 'rally', 'rallies', 'rallying',
    'jump', 'jumps', 'jumping', 'gain', 'gains', 'gaining', 'climb', 'climbs', 'climbing',
    'rise', 'rises', 'rising', 'boost', 'boosts', 'boosting', 'spike', 'spikes', 'spiking',

    # Moderate positive
    'growth', 'growing', 'grew', 'increase', 'increases', 'increasing', 'improved', 'improvement',
    'strong', 'stronger', 'strongest', 'positive', 'optimistic', 'optimism', 'bullish', 'bull',
    'beat', 'beats', 'beating', 'exceed', 'exceeds', 'exceeded', 'outperform', 'outperforms',

    # Mild positive
    'up', 'higher', 'advance', 'advances', 'advancing', 'recovery', 'recover', 'recovers',
    'rebound', 'rebounds', 'rebounding', 'momentum', 'confidence', 'confident',
    'success', 'successful', 'win', 'wins', 'winning', 'profit', 'profits', 'profitable'
}

NEGATIVE_KEYWORDS = {
    # Strong negative
    'plunge', 'plunges', 'plunging', 'crash', 'crashes', 'crashing', 'collapse', 'collapses',
    'tumble', 'tumbles', 'tumbling', 'slump', 'slumps', 'slumping', 'tank', 'tanks', 'tanking',

    # Moderate negative
    'fall', 'falls', 'falling', 'fell', 'drop', 'drops', 'dropping', 'dropped', 'decline',
    'declines', 'declining', 'declined', 'loss', 'losses', 'losing', 'lost', 'negative',
    'pessimistic', 'pessimism', 'bearish', 'bear', 'weak', 'weaker', 'weakest', 'weakness',

    # Mild negative
    'down', 'lower', 'slip', 'slips', 'slipping', 'slipped', 'concern', 'concerns', 'concerned',
    'worry', 'worries', 'worried', 'fear', 'fears', 'fearing', 'risk', 'risks', 'risky',
    'uncertainty', 'uncertain', 'volatile', 'volatility', 'struggle', 'struggles', 'struggling'
}

NEUTRAL_KEYWORDS = {
    'flat', 'unchanged', 'steady', 'stable', 'mixed', 'await', 'awaits', 'awaiting',
    'expect', 'expects', 'expected', 'watch', 'watches', 'watching', 'monitor', 'monitors',
    'consolidate', 'consolidates', 'consolidating', 'pause', 'pauses', 'pausing'
}


def calculate_sentiment(text: str) -> float:
    """Calculate sentiment score from text.

    Args:
        text: Text to analyze (headline + body)

    Returns:
        Sentiment score from -1.0 to 1.0
    """
    text_lower = text.lower()

    # Count keyword matches
    positive_count = sum(1 for word in POSITIVE_KEYWORDS if re.search(r'\b' + word + r'\b', text_lower))
    negative_count = sum(1 for word in NEGATIVE_KEYWORDS if re.search(r'\b' + word + r'\b', text_lower))
    neutral_count = sum(1 for word in NEUTRAL_KEYWORDS if re.search(r'\b' + word + r'\b', text_lower))

    # Weight headline more heavily
    headline_lower = text_lower.split('.')[0]  # First sentence approximation
    headline_pos = sum(1 for word in POSITIVE_KEYWORDS if re.search(r'\b' + word + r'\b', headline_lower))
    headline_neg = sum(1 for word in NEGATIVE_KEYWORDS if re.search(r'\b' + word + r'\b', headline_lower))

    # Apply headline weighting
    positive_count += headline_pos * 2
    negative_count += headline_neg * 2

    total = positive_count + negative_count

    if total == 0:
        # No strong sentiment keywords found
        return 0.0

    # Calculate base sentiment
    sentiment = (positive_count - negative_count) / (total + neutral_count + 1)

    # Normalize to -1 to 1 range with some dampening
    sentiment = max(-1.0, min(1.0, sentiment * 1.5))

    return round(sentiment, 3)


def classify_sentiment_category(score: float) -> str:
    """Classify sentiment score into category.

    Args:
        score: Sentiment score from -1.0 to 1.0

    Returns:
        'positive', 'negative', or 'neutral'
    """
    if score > 0.2:
        return 'positive'
    elif score < -0.2:
        return 'negative'
    else:
        return 'neutral'


def process_article(article: Dict) -> Dict:
    """Add sentiment score to an article.

    Args:
        article: Article dictionary

    Returns:
        Article with sentiment_score added
    """
    # Combine headline and body for analysis
    text = f"{article.get('headline', '')} {article.get('body', '')}"

    # Calculate sentiment
    sentiment_score = calculate_sentiment(text)

    # Add to article
    article['sentiment_score'] = sentiment_score
    article['sentiment_category'] = classify_sentiment_category(sentiment_score)

    return article


def process_articles(input_dir: Path, output_dir: Path, dry_run: bool = False):
    """Process all articles in directory and add sentiment scores.

    Args:
        input_dir: Directory containing article JSON files
        output_dir: Directory to write updated articles
        dry_run: If True, only print statistics without writing
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }

    articles_by_sentiment = {
        'positive': [],
        'negative': [],
        'neutral': []
    }

    print(f"Processing articles from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Process each JSON file
    for json_file in sorted(input_dir.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                article = json.load(f)

            # Add sentiment
            article = process_article(article)

            # Update statistics
            stats['total'] += 1
            category = article['sentiment_category']
            stats[category] += 1

            # Store for preview
            articles_by_sentiment[category].append({
                'headline': article['headline'],
                'score': article['sentiment_score'],
                'file': json_file.name
            })

            # Write to output
            if not dry_run:
                output_file = output_dir / json_file.name
                with open(output_file, 'w') as f:
                    json.dump(article, f, indent=2, ensure_ascii=False)

            # Print progress
            emoji = "🟢" if category == 'positive' else "🔴" if category == 'negative' else "⚪"
            print(f"{emoji} {article['sentiment_score']:+.3f} | {article['headline'][:70]}")

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total articles: {stats['total']}")
    print(f"  🟢 Positive: {stats['positive']} ({stats['positive']/stats['total']*100:.1f}%)")
    print(f"  🔴 Negative: {stats['negative']} ({stats['negative']/stats['total']*100:.1f}%)")
    print(f"  ⚪ Neutral:  {stats['neutral']} ({stats['neutral']/stats['total']*100:.1f}%)")

    # Show examples
    print("\n" + "=" * 60)
    print("EXAMPLES BY CATEGORY")
    print("=" * 60)

    for category in ['positive', 'negative', 'neutral']:
        emoji = "🟢" if category == 'positive' else "🔴" if category == 'negative' else "⚪"
        print(f"\n{emoji} {category.upper()} (showing up to 3):")
        for article in articles_by_sentiment[category][:3]:
            print(f"  {article['score']:+.3f} | {article['headline'][:70]}")

    if dry_run:
        print("\n⚠️  DRY RUN - No files were written")
    else:
        print(f"\n✓ Processed {stats['total']} articles")
        print(f"✓ Written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/news/bronze/raw_articles'),
        help='Directory containing articles to process'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/news/bronze/raw_articles'),
        help='Directory to write updated articles (default: overwrites input)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview sentiment classification without writing files'
    )

    args = parser.parse_args()

    process_articles(args.input_dir, args.output_dir, args.dry_run)


if __name__ == '__main__':
    main()
