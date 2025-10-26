#!/usr/bin/env python3
"""News Simulator Flask Application.

A simple web interface with 3 buttons to stream positive, negative,
or neutral news articles to the Bronze pipeline.
"""

import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
NEWS_DATA_DIR = Path("../data/news/bronze/raw_articles/")
NEWS_OUTPUT_DIR = Path("../data/news/bronze/simulated/")
NEWS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage for loaded news
news_database = {
    'positive': [],
    'negative': [],
    'neutral': [],
    'all': []
}

# Statistics
stats = {
    'total_articles': 0,
    'streamed_count': 0,
    'last_article': None
}


def load_news_articles():
    """Load all news articles from Bronze directory."""
    logger.info(f"Loading news articles from {NEWS_DATA_DIR}")

    if not NEWS_DATA_DIR.exists():
        logger.warning(f"News directory not found: {NEWS_DATA_DIR}")
        return

    article_count = 0

    # Load from individual JSON files
    for json_file in NEWS_DATA_DIR.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                article = json.load(f)
                news_database['all'].append(article)
                article_count += 1

                # Classify by sentiment if available
                sentiment = article.get('sentiment_score', 0.0)
                if sentiment > 0.2:
                    news_database['positive'].append(article)
                elif sentiment < -0.2:
                    news_database['negative'].append(article)
                else:
                    news_database['neutral'].append(article)

        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    # Also load from NDJSON files
    for ndjson_file in NEWS_DATA_DIR.glob("*.ndjson"):
        try:
            with open(ndjson_file, 'r') as f:
                for line in f:
                    if line.strip():
                        article = json.loads(line)
                        news_database['all'].append(article)
                        article_count += 1

                        sentiment = article.get('sentiment_score', 0.0)
                        if sentiment > 0.2:
                            news_database['positive'].append(article)
                        elif sentiment < -0.2:
                            news_database['negative'].append(article)
                        else:
                            news_database['neutral'].append(article)

        except Exception as e:
            logger.error(f"Error loading {ndjson_file}: {e}")

    stats['total_articles'] = article_count

    logger.info(f"Loaded {article_count} articles")
    logger.info(f"  Positive: {len(news_database['positive'])}")
    logger.info(f"  Negative: {len(news_database['negative'])}")
    logger.info(f"  Neutral: {len(news_database['neutral'])}")


def calculate_sentiment(text: str) -> float:
    """Simple sentiment calculation based on keywords.

    Args:
        text: Article text

    Returns:
        Sentiment score from -1.0 to 1.0
    """
    text_lower = text.lower()

    positive_words = ['gain', 'surge', 'rise', 'boost', 'positive', 'growth', 'rally', 'bullish', 'up']
    negative_words = ['fall', 'drop', 'decline', 'loss', 'negative', 'crash', 'bearish', 'down', 'plunge']

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def create_mock_article(sentiment_type: str) -> Dict:
    """Create a mock news article when database is empty.

    Args:
        sentiment_type: "positive", "negative", or "neutral"

    Returns:
        Mock article dictionary
    """
    templates = {
        'positive': [
            "S&P 500 surges to new highs on strong earnings reports",
            "Tech stocks rally as market sentiment improves",
            "Federal Reserve signals continued economic growth",
            "Corporate earnings beat expectations across sectors",
            "Bull market extends as investors remain optimistic"
        ],
        'negative': [
            "S&P 500 declines amid economic uncertainty",
            "Market sell-off continues as investors flee to safety",
            "Federal Reserve raises concerns about inflation",
            "Tech stocks plunge on weak guidance",
            "Bear market fears resurface amid volatility"
        ],
        'neutral': [
            "S&P 500 closes flat in quiet trading session",
            "Markets await Federal Reserve policy decision",
            "Mixed signals from economic data keep investors cautious",
            "Traders take profits after recent rally",
            "Market consolidates at current levels"
        ]
    }

    headline = random.choice(templates[sentiment_type])

    sentiment_scores = {
        'positive': random.uniform(0.5, 0.9),
        'negative': random.uniform(-0.9, -0.5),
        'neutral': random.uniform(-0.2, 0.2)
    }

    return {
        'headline': headline,
        'published_at': datetime.now().isoformat(),
        'source': 'MockNewsAPI',
        'sentiment_score': sentiment_scores[sentiment_type],
        'url': f'https://mock-news.com/article-{random.randint(1000, 9999)}',
        'body': f'{headline}. Market analysis continues to develop.',
        'mock': True
    }


@app.route('/')
def index():
    """Render main UI."""
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    """Get current statistics."""
    return jsonify({
        'total_articles': stats['total_articles'],
        'positive': len(news_database['positive']),
        'negative': len(news_database['negative']),
        'neutral': len(news_database['neutral']),
        'streamed_count': stats['streamed_count'],
        'last_article': stats['last_article']
    })


@app.route('/api/stream/<sentiment_type>', methods=['POST'])
def stream_news(sentiment_type: str):
    """Stream a news article of specified sentiment type.

    Args:
        sentiment_type: "positive", "negative", or "neutral"

    Returns:
        JSON response with streamed article
    """
    if sentiment_type not in ['positive', 'negative', 'neutral']:
        return jsonify({'error': 'Invalid sentiment type'}), 400

    # Get articles of specified sentiment
    available_articles = news_database.get(sentiment_type, [])

    if not available_articles:
        logger.warning(f"No {sentiment_type} articles available, creating mock article")
        article = create_mock_article(sentiment_type)
        logger.info(f"Created mock {sentiment_type} article: {article['headline']}")
    else:
        # Pick random article and make a copy to avoid modifying the original
        article = random.choice(available_articles).copy()
        logger.info(f"Selected existing {sentiment_type} article: {article.get('headline', 'No headline')}")

    # Add streaming metadata
    article['streamed_at'] = datetime.now().isoformat()
    article['sentiment_type'] = sentiment_type

    # Save to simulated stream
    output_file = NEWS_OUTPUT_DIR / f"simulated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(article, f, indent=2)

        logger.info(f"Streamed {sentiment_type} article: {article.get('headline', 'No headline')}")

        # Update statistics
        stats['streamed_count'] += 1
        stats['last_article'] = {
            'time': article['streamed_at'],
            'headline': article.get('headline', 'No headline'),
            'sentiment': article.get('sentiment_score', 0.0),
            'type': sentiment_type
        }

        return jsonify({
            'status': 'success',
            'article': {
                'headline': article.get('headline', 'No headline'),
                'published_at': article.get('published_at', ''),
                'source': article.get('source', 'Unknown'),
                'sentiment_score': article.get('sentiment_score', 0.0),
                'sentiment_type': sentiment_type,
                'streamed_at': article['streamed_at']
            },
            'output_file': str(output_file)
        })

    except Exception as e:
        logger.error(f"Error streaming article: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reload', methods=['POST'])
def reload_articles():
    """Reload articles from disk."""
    try:
        load_news_articles()
        return jsonify({
            'status': 'success',
            'total_articles': stats['total_articles']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Load articles on startup
load_news_articles()


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
