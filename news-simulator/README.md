# 📰 News Simulator

A simple web interface for streaming positive, negative, or neutral news articles to the Bronze pipeline.

## Features

- **3-Button Interface**: Stream positive, negative, or neutral news with a single click
- **Real-time Statistics**: See available articles and streaming counts
- **Article Preview**: View the last streamed article with sentiment analysis
- **Auto-reload**: Automatically refresh statistics every 5 seconds
- **Mock Articles**: Generates mock news when database is empty

## Quick Start

### Option 1: Local Python

```bash
# Install dependencies
pip install flask flask-cors

# Run the simulator
cd news-simulator
python app.py
```

Visit: http://localhost:5000

### Option 2: Docker

```bash
# Build and run
docker build -t news-simulator .
docker run -p 5000:5000 -v $(pwd)/data:/app/data news-simulator
```

## Usage

1. **Load Articles**: The simulator automatically loads news from `../data/news/bronze/raw_articles/`
2. **Stream News**: Click one of the three buttons:
   - 🟢 **Positive**: Stream news with positive sentiment
   - 🔴 **Negative**: Stream news with negative sentiment
   - ⚪ **Neutral**: Stream news with neutral sentiment
3. **View Results**: See the streamed article appear below the buttons
4. **Reload**: Click "🔄 Reload Articles" to refresh the article database

## API Endpoints

### GET `/api/stats`
Get current statistics

**Response:**
```json
{
  "total_articles": 1247,
  "positive": 450,
  "negative": 380,
  "neutral": 417,
  "streamed_count": 42,
  "last_article": {
    "time": "2025-10-13T10:30:00",
    "headline": "S&P 500 surges...",
    "sentiment": 0.75,
    "type": "positive"
  }
}
```

### POST `/api/stream/{sentiment_type}`
Stream an article of specified sentiment

**Parameters:**
- `sentiment_type`: "positive", "negative", or "neutral"

**Response:**
```json
{
  "status": "success",
  "article": {
    "headline": "...",
    "sentiment_score": 0.75,
    "streamed_at": "2025-10-13T10:30:00"
  }
}
```

### POST `/api/reload`
Reload articles from disk

## Output

Streamed articles are saved to:
```
../data/news/bronze/simulated/simulated_YYYYMMDD_HHMMSS.json
```

## Sentiment Classification

Articles are classified based on `sentiment_score`:
- **Positive**: score > 0.2
- **Negative**: score < -0.2
- **Neutral**: -0.2 ≤ score ≤ 0.2

## Mock Articles

If no real articles are available, the simulator generates mock news:

- **Positive**: "S&P 500 surges to new highs..."
- **Negative**: "S&P 500 declines amid uncertainty..."
- **Neutral**: "S&P 500 closes flat in quiet session..."

## Development

### Project Structure

```
news-simulator/
├── app.py                  # Flask backend
├── templates/
│   └── index.html         # Main UI
├── static/
│   ├── style.css          # Styling
│   └── app.js             # Frontend logic
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

### Environment Variables

- `NEWS_DATA_DIR`: Directory containing news articles (default: `../data/news/bronze/raw_articles/`)
- `NEWS_OUTPUT_DIR`: Output directory for streamed articles (default: `../data/news/bronze/simulated/`)

## Integration with Pipeline

To integrate streamed news into the Bronze pipeline:

1. **Monitor Simulated Directory**: Watch `data/news/bronze/simulated/` for new files
2. **Process Articles**: Feed them into the Bronze → Silver → Gold pipeline
3. **Feature Extraction**: Extract sentiment, entities, and topics
4. **Generate Signals**: Create trading signals from news

## Troubleshooting

**No articles available**
- Check that `../data/news/bronze/raw_articles/` exists
- Run the news scraper first to populate articles
- The simulator will generate mock articles as fallback

**Connection refused**
- Ensure Flask is running on port 5000
- Check firewall settings
- Verify correct API_BASE URL in app.js

**CORS errors**
- Flask-CORS is enabled by default
- For production, configure `allow_origins` in app.py

## License

Part of the S&P 500 ML Prediction Pipeline project.
