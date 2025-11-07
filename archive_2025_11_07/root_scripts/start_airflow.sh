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

# Create airflow_mlops structure if needed
if [ ! -d "airflow_mlops" ]; then
    echo "📁 Creating airflow_mlops directory..."
    mkdir -p airflow_mlops/dags
    cp -r docker/airflow/dags/* airflow_mlops/dags/ 2>/dev/null || true

    cat > airflow_mlops/init-airflow.sh << 'EOF'
#!/bin/bash
set -e
echo "Initializing Airflow database..."
airflow db init
echo "Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
echo "Airflow initialization complete!"
EOF
    chmod +x airflow_mlops/init-airflow.sh
    echo "✅ Created airflow_mlops directory"
fi

# Build images if needed
echo ""
echo "🔨 Building Docker images (this may take a few minutes)..."
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

# Start Airflow services
echo ""
echo "🚁 Starting Airflow services..."
docker-compose up -d airflow-webserver airflow-scheduler airflow-dag-processor

echo "⏳ Waiting for Airflow to start (this may take 1-2 minutes)..."
sleep 60

# Check status
echo ""
echo "📊 Final Service Status:"
docker-compose ps

echo ""
echo "========================================="
echo "✅ Airflow is running!"
echo "========================================="
echo ""
echo "🌐 Airflow UI:  http://localhost:8080"
echo "👤 Username:    admin"
echo "🔑 Password:    admin"
echo ""
echo "📊 MLflow UI:   http://localhost:5000"
echo ""
echo "📝 Available DAGs:"
echo "   • sp500_ml_pipeline_v2"
echo "   • sp500_ml_pipeline_v3_production"
echo ""
echo "📚 Commands:"
echo "   View logs:        docker-compose logs -f airflow-scheduler"
echo "   Stop services:    docker-compose down"
echo "   Restart:          docker-compose restart airflow-scheduler"
echo ""
echo "🎯 Next Steps:"
echo "   1. Open http://localhost:8080 and login"
echo "   2. Enable a DAG by toggling the switch"
echo "   3. Click the ▶️ button to trigger manually"
echo ""
