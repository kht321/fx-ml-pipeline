#!/bin/bash
# Daily News Collection Cron Job Setup
# This script helps you set up automated news collection every 6 hours

PROJECT_DIR="/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
NEWS_SCRIPT="$PROJECT_DIR/src_clean/data_pipelines/bronze/news_data_collector.py"

# Cron job command
CRON_CMD="0 */6 * * * cd \"$PROJECT_DIR\" && \"$PYTHON_BIN\" \"$NEWS_SCRIPT\" --mode recent >> \"$PROJECT_DIR/logs/news_collection.log\" 2>&1"

echo "================================================================"
echo "Daily News Collection Setup"
echo "================================================================"
echo ""
echo "This will set up automated news collection every 6 hours."
echo ""
echo "Cron command:"
echo "$CRON_CMD"
echo ""
echo "To install this cron job:"
echo "1. Open crontab editor: crontab -e"
echo "2. Add the following line:"
echo ""
echo "$CRON_CMD"
echo ""
echo "3. Save and exit"
echo ""
echo "To verify:"
echo "  crontab -l"
echo ""
echo "To check logs:"
echo "  tail -f $PROJECT_DIR/logs/news_collection.log"
echo ""
echo "================================================================"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Test the command once
echo "Testing news collection command..."
cd "$PROJECT_DIR" && "$PYTHON_BIN" "$NEWS_SCRIPT" --mode recent

echo ""
echo "If the test above succeeded, you can now install the cron job."
