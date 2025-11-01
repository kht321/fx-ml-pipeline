#!/bin/bash
set -e

echo "🚀 Starting FX-ML-Pipeline Airflow"
echo "==================================="

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running"

# Create .env if doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << 'ENVEOF'
# Docker paths
HOST_DATA_DIR=/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/data_clean
HOST_MODELS_DIR=/Users/kevintaukoor/Projects/MLE Group Original/fx-ml-pipeline/data_clean/models

# OANDA (optional - for live data)
OANDA_ACCOUNT_ID=
OANDA_TOKEN=
OANDA_ENV=practice

# News API keys (optional)
NEWS_API_KEY=
ENVEOF
    echo "✅ Created .env file"
fi

# Create airflow_mlops structure if needed
if [ ! -d "airflow_mlops" ]; then
    echo "📁 Creating airflow_mlops directory..."
    mkdir -p airflow_mlops/dags

    # Copy DAGs if they exist
    if [ -d "docker/airflow/dags" ]; then
        cp -r docker/airflow/dags/* airflow_mlops/dags/ 2>/dev/null || true
        echo "✅ Copied DAG files"
    fi

    cat > airflow_mlops/init-airflow.sh << 'EOF'
#!/bin/bash
set -e
echo "Initializing Airflow database..."
airflow db migrate

echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "User already exists"

echo "Airflow initialization complete!"
EOF
    chmod +x airflow_mlops/init-airflow.sh
    echo "✅ Created airflow_mlops directory"
fi

# Build images if needed
echo ""
echo "🔨 Building Docker images (this may take 5-10 minutes first time)..."
docker-compose build airflow-init

# Start infrastructure
echo ""
echo "🏗️  Starting infrastructure services (postgres, redis, mlflow)..."
docker-compose up -d postgres redis mlflow

echo "⏳ Waiting 30 seconds for infrastructure to initialize..."
sleep 30

# Check infrastructure
echo "📊 Checking infrastructure status..."
docker-compose ps postgres redis mlflow

# Initialize Airflow
echo ""
echo "🔧 Initializing Airflow database (first time only)..."
docker-compose up airflow-init

# Wait a bit
echo "⏳ Waiting 10 seconds..."
sleep 10

# Start Airflow services
echo ""
echo "🚁 Starting Airflow services..."
docker-compose up -d airflow-webserver airflow-scheduler airflow-dag-processor

echo "⏳ Waiting for Airflow to start (this may take 1-2 minutes)..."
echo "   You can check progress with: docker-compose logs -f airflow-webserver"
sleep 90

# Check status
echo ""
echo "📊 Final Service Status:"
docker-compose ps | grep -E "postgres|redis|mlflow|airflow"

echo ""
echo "========================================="
echo "✅ Airflow should be running!"
echo "========================================="
echo ""
echo "🌐 Airflow UI:  http://localhost:8080"
echo "👤 Username:    admin"
echo "🔑 Password:    admin"
echo ""
echo "📊 MLflow UI:   http://localhost:5001"
echo ""
echo "📝 Available DAGs:"
echo "   • sp500_ml_pipeline_v2"
echo "   • sp500_ml_pipeline_v3_production"
echo ""
echo "📚 Useful Commands:"
echo "   View logs:        docker-compose logs -f airflow-scheduler"
echo "   Stop services:    docker-compose down"
echo "   Restart:          docker-compose restart airflow-scheduler"
echo "   Check status:     docker-compose ps"
echo ""
echo "🎯 Next Steps:"
echo "   1. Wait 1-2 more minutes for Airflow to fully start"
echo "   2. Open http://localhost:8080 and login"
echo "   3. Enable a DAG by toggling the switch"
echo "   4. Click the ▶️ button to trigger manually"
echo ""
echo "⚠️  If you see errors, check logs with:"
echo "   docker-compose logs airflow-webserver"
echo "   docker-compose logs airflow-scheduler"
echo ""
