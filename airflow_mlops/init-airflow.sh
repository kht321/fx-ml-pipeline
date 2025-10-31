#!/bin/bash
set -e

echo "🔧 Initializing Airflow with standalone mode (workaround for SQLAlchemy issues)..."

# Use airflow standalone init which handles database setup more gracefully
airflow db init || airflow db migrate || echo "DB already initialized"

echo "👤 Creating admin user..."
# Create user with simple command
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin 2>/dev/null || echo "ℹ️  User already exists"

echo "✅ Airflow initialization complete!"
