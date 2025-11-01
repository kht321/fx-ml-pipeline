#!/bin/bash
set -e

echo "🔧 Fixing Port 5000 Conflict"
echo "=============================="

# Check if port 5000 is in use
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 5000 is in use (likely macOS ControlCenter)"
    echo "✅ MLflow has been reconfigured to use port 5001"
    echo ""
fi

echo "🚀 Starting services with MLflow on port 5001..."
echo ""

# Stop any existing containers
echo "🛑 Stopping any existing containers..."
docker-compose down 2>/dev/null || true

# Start the fixed script
./start_airflow_fixed.sh

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "📝 IMPORTANT: MLflow is now on port 5001"
echo ""
echo "🌐 Access Points:"
echo "   • Airflow UI:  http://localhost:8080"
echo "   • MLflow UI:   http://localhost:5001  ⚠️ NEW PORT!"
echo ""
echo "👤 Login: admin / admin"
echo ""
