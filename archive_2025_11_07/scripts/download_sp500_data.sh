#!/bin/bash
# Convenience script to download S&P 500 historical data
# This script activates the virtual environment and runs the download

set -e  # Exit on error

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "================================================"
echo "S&P 500 Historical Data Download"
echo "================================================"
echo ""

# Check if .venv exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "ERROR: Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if .env exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "ERROR: .env file not found"
    echo "Please create a .env file with your OANDA credentials:"
    echo "  OANDA_TOKEN=your_token_here"
    echo "  OANDA_ENV=practice"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Change to project root
cd "$PROJECT_ROOT"

# Default parameters
GRANULARITY="M1"
YEARS="10"
RATE_LIMIT="0.5"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --granularity)
            GRANULARITY="$2"
            shift 2
            ;;
        --years)
            YEARS="$2"
            shift 2
            ;;
        --rate-limit-delay)
            RATE_LIMIT="$2"
            shift 2
            ;;
        --validate-only)
            VALIDATE_ONLY="--validate-only"
            shift
            ;;
        --hourly)
            GRANULARITY="H1"
            echo "Using hourly granularity (better for 10 years of data)"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --granularity GRAN    Candle granularity (M1, M5, H1, etc.)"
            echo "  --years NUM           Number of years to download (default: 10)"
            echo "  --rate-limit-delay N  Delay between requests in seconds (default: 0.5)"
            echo "  --validate-only       Only validate existing data"
            echo "  --hourly              Shortcut for --granularity H1"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Download 10 years of 1-minute data"
            echo "  $0 --hourly           # Download 10 years of hourly data"
            echo "  $0 --years 5          # Download 5 years of 1-minute data"
            echo "  $0 --validate-only    # Validate existing data"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python src/download_sp500_historical.py"
CMD="$CMD --granularity $GRANULARITY"
CMD="$CMD --years $YEARS"
CMD="$CMD --rate-limit-delay $RATE_LIMIT"

if [ ! -z "$VALIDATE_ONLY" ]; then
    CMD="$CMD $VALIDATE_ONLY"
fi

# Show what we're doing
echo ""
echo "Configuration:"
echo "  Instrument:  SPX500_USD (S&P 500)"
echo "  Granularity: $GRANULARITY"
echo "  Years:       $YEARS"
echo "  Rate Limit:  ${RATE_LIMIT}s between requests"
echo ""

if [ ! -z "$VALIDATE_ONLY" ]; then
    echo "Mode: Validation only"
else
    echo "Mode: Download"
    echo ""
    echo "NOTE: OANDA typically has limited historical data availability:"
    echo "  - 1-minute (M1): ~1-2 years"
    echo "  - Hourly (H1):   ~4-5 years"
    echo "  - Daily (D):     10+ years"
    echo ""
    echo "For 10 years of data, consider using --hourly for hourly candles."
    echo ""
    read -p "Press Enter to continue or Ctrl+C to cancel..."
fi

echo ""
echo "================================================"
echo "Starting download..."
echo "================================================"
echo ""

# Run the download
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✓ Success!"
    echo "================================================"
    echo ""
    echo "Data saved to: data/bronze/prices/"
    echo ""
    echo "Next steps:"
    echo "  1. Validate data: $0 --validate-only"
    echo "  2. Process data: Use existing pipeline scripts"
    echo "  3. Train models: Use processed features for training"
else
    echo ""
    echo "================================================"
    echo "✗ Download failed or interrupted"
    echo "================================================"
    echo ""
    echo "To resume: $0 (progress was saved)"
fi
