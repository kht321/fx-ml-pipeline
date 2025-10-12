#!/bin/bash
# Setup script for news API keys

echo "=================================================================================="
echo "S&P 500 News Scraper - API Key Setup"
echo "=================================================================================="
echo ""
echo "This script will help you add API keys to your .env file."
echo ""
echo "📋 Free API Key Sources:"
echo "   1. NewsAPI:       https://newsapi.org/register"
echo "   2. Alpha Vantage: https://www.alphavantage.co/support/#api-key"
echo "   3. Finnhub:       https://finnhub.io/register"
echo ""
echo "All offer FREE tiers with no credit card required!"
echo ""
echo "=================================================================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
fi

# Function to add or update key in .env
add_key() {
    local key_name=$1
    local key_value=$2

    if grep -q "^${key_name}=" .env; then
        # Update existing key
        sed -i.bak "s|^${key_name}=.*|${key_name}=${key_value}|" .env
        echo "✓ Updated ${key_name}"
    else
        # Add new key
        echo "${key_name}=${key_value}" >> .env
        echo "✓ Added ${key_name}"
    fi
}

echo "Let's add your API keys (press Enter to skip any):"
echo ""

# NewsAPI
read -p "Enter NewsAPI key (or press Enter to skip): " newsapi_key
if [ ! -z "$newsapi_key" ]; then
    add_key "NEWSAPI_KEY" "$newsapi_key"
fi

# Alpha Vantage
read -p "Enter Alpha Vantage key (or press Enter to skip): " av_key
if [ ! -z "$av_key" ]; then
    add_key "ALPHAVANTAGE_KEY" "$av_key"
fi

# Finnhub
read -p "Enter Finnhub key (or press Enter to skip): " finnhub_key
if [ ! -z "$finnhub_key" ]; then
    add_key "FINNHUB_KEY" "$finnhub_key"
fi

echo ""
echo "=================================================================================="
echo "✓ Setup complete!"
echo ""
echo "📊 API Capabilities:"
echo "   • NewsAPI:       Last 30 days (free tier)"
echo "   • Alpha Vantage: Recent news + sentiment"
echo "   • Finnhub:       Real-time + recent news"
echo ""
echo "💡 For full 5-year history, see docs/NEWS_SCRAPING_GUIDE.md for alternatives."
echo ""
echo "🚀 Ready to scrape! Run:"
echo "   python3 src/scrape_historical_sp500_news.py --recent-only"
echo "=================================================================================="
