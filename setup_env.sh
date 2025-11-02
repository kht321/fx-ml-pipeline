#!/bin/bash
# FX ML Pipeline - Automatic .env Setup for Mac/Linux
# This script creates a .env file with your current project path

echo "========================================"
echo "FX ML Pipeline - Environment Setup"
echo "========================================"
echo ""

# Get the current directory (project root)
CURRENT_DIR="$(pwd)"

echo "Detected project root: $CURRENT_DIR"
echo ""

# Check if .env already exists
if [ -f .env ]; then
    echo "WARNING: .env file already exists!"
    read -p "Do you want to overwrite it? (y/n): " OVERWRITE
    if [ "$OVERWRITE" != "y" ] && [ "$OVERWRITE" != "Y" ]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# Create .env file
echo "Creating .env file..."
cat > .env << EOF
# FX ML Pipeline Environment Variables
# Auto-generated on $(date)

# Project root - already using forward slashes
FX_ML_PIPELINE_ROOT=$CURRENT_DIR

# Derived paths
HOST_DATA_DIR=\${FX_ML_PIPELINE_ROOT}/data_clean
HOST_MODELS_DIR=\${FX_ML_PIPELINE_ROOT}/data_clean/models

# OANDA API Credentials (replace with your actual credentials)
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_TOKEN=your_api_token_here
EOF

echo ""
echo "========================================"
echo "SUCCESS! .env file created"
echo "========================================"
echo ""
echo "Project root set to: $CURRENT_DIR"
echo ""
echo "Next steps:"
echo "1. Update OANDA credentials in .env if needed"
echo "2. Run: docker-compose up -d"
echo ""
echo "For more information, see SETUP_ENVIRONMENT.md"
echo ""
