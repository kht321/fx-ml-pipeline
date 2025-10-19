#!/bin/bash

# Setup script for Hybrid News Scraper
# Configures free API keys and tests the scraper

set -e  # Exit on error

echo "============================================================================"
echo "HYBRID NEWS SCRAPER SETUP - FREE 5-YEAR COLLECTION"
echo "============================================================================"
echo ""

# ============================================================================
# 1. Check Python dependencies
# ============================================================================

echo "Step 1: Checking Python dependencies..."
echo ""

REQUIRED_PACKAGES="requests aiohttp beautifulsoup4 feedparser python-dateutil pytz"

for package in $REQUIRED_PACKAGES; do
    if python3 -c "import ${package//-/_}" 2>/dev/null; then
        echo "✓ $package installed"
    else
        echo "✗ $package NOT installed"
        echo "  Installing $package..."
        pip install $package
    fi
done

echo ""

# ============================================================================
# 2. Setup API keys (optional but recommended)
# ============================================================================

echo "Step 2: API Key Setup (Optional - Free Tiers)"
echo ""
echo "The scraper works without API keys (using GDELT), but API keys provide"
echo "additional coverage and more recent news."
echo ""

ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file..."
    touch "$ENV_FILE"
fi

# Alpha Vantage
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Alpha Vantage (FREE - 25 calls/day)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Signup: https://www.alphavantage.co/support/#api-key"
echo ""

if grep -q "ALPHAVANTAGE_KEY=" "$ENV_FILE"; then
    echo "✓ Alpha Vantage key already configured"
else
    read -p "Enter your Alpha Vantage API key (or press Enter to skip): " AV_KEY
    if [ -n "$AV_KEY" ]; then
        echo "ALPHAVANTAGE_KEY=$AV_KEY" >> "$ENV_FILE"
        echo "✓ Alpha Vantage key saved to .env"
    else
        echo "⊘ Skipped Alpha Vantage (GDELT will still work)"
    fi
fi

echo ""

# Finnhub
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Finnhub (FREE - 60 calls/min, 1 year history)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Signup: https://finnhub.io/register"
echo ""

if grep -q "FINNHUB_KEY=" "$ENV_FILE"; then
    echo "✓ Finnhub key already configured"
else
    read -p "Enter your Finnhub API key (or press Enter to skip): " FH_KEY
    if [ -n "$FH_KEY" ]; then
        echo "FINNHUB_KEY=$FH_KEY" >> "$ENV_FILE"
        echo "✓ Finnhub key saved to .env"
    else
        echo "⊘ Skipped Finnhub (GDELT will still work)"
    fi
fi

echo ""

# ============================================================================
# 3. Create output directories
# ============================================================================

echo "Step 3: Creating output directories..."
echo ""

mkdir -p data_clean/bronze/news/hybrid
echo "✓ Created: data_clean/bronze/news/hybrid/"

echo ""

# ============================================================================
# 4. Test the scraper
# ============================================================================

echo "Step 4: Testing the scraper..."
echo ""

echo "Running a test collection for yesterday's news..."
echo ""

python3 src_clean/data_pipelines/bronze/hybrid_news_scraper.py \
    --mode incremental \
    --sources gdelt \
    --output-dir data_clean/bronze/news/hybrid

echo ""

# ============================================================================
# 5. Setup instructions
# ============================================================================

echo "============================================================================"
echo "SETUP COMPLETE!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Collect 5 years of news (2020-2025):"
echo "   python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \\"
echo "     --start-date 2020-10-19 \\"
echo "     --end-date 2025-10-19 \\"
echo "     --sources gdelt"
echo ""
echo "2. Collect recent news (past year) with Finnhub:"
echo "   python src_clean/data_pipelines/bronze/hybrid_news_scraper.py \\"
echo "     --start-date 2024-10-19 \\"
echo "     --sources finnhub"
echo ""
echo "3. Setup daily cron job (optional):"
echo "   crontab -e"
echo "   # Add this line:"
echo "   0 1 * * * cd $(pwd) && python3 src_clean/data_pipelines/bronze/hybrid_news_scraper.py --mode incremental --sources all"
echo ""
echo "4. View configuration:"
echo "   cat configs/hybrid_news_sources.yaml"
echo ""
echo "============================================================================"
echo ""
echo "FREE SOURCES AVAILABLE:"
echo "  ✓ GDELT (2017-present, unlimited)"
echo "  ✓ Alpha Vantage (25 calls/day) - $(grep -q 'ALPHAVANTAGE_KEY=' .env && echo 'CONFIGURED' || echo 'NOT CONFIGURED')"
echo "  ✓ Finnhub (60 calls/min, 1yr) - $(grep -q 'FINNHUB_KEY=' .env && echo 'CONFIGURED' || echo 'NOT CONFIGURED')"
echo ""
echo "Expected collection for 2017-2025:"
echo "  ~50,000 - 100,000 articles"
echo "  2-5 hours runtime"
echo "  $0 cost"
echo ""
echo "============================================================================"
