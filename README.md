# S&P 500 ML Prediction Pipeline

End-to-end machine learning pipeline for S&P 500 price prediction using technical indicators and news sentiment.

## 🎯 Status

✅ **Production Ready** - Full pipeline operational with Python 3.11

## 🚀 Quick Demo (5 minutes)

```bash
# 1. Setup Python 3.11 environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start MLflow tracking
mlflow server --host 0.0.0.0 --port 5000 &

# 3. Train model
python src_clean/training/xgboost_training_pipeline_mlflow.py \
  --market-features data_clean/gold/market/features/spx500_features.csv \
  --news-signals data_clean/gold/news/signals/sp500_trading_signals.csv \
  --prediction-horizon 30 \
  --mlflow-uri http://localhost:5000

# 4. Launch dashboard
streamlit run src_clean/ui/streamlit_dashboard.py &

# 5. Access services
# - Streamlit Dashboard: http://localhost:8501
# - MLflow UI: http://localhost:5000
```

## 📚 Complete System Demo

**For comprehensive walkthrough**, see **[Complete Demo Guide](docs/COMPLETE_DEMO_GUIDE.md)**

30-minute demo includes:
- ✅ Data ingestion (Market + News simulator)
- ✅ Feature engineering (Bronze → Silver → Gold)
- ✅ Model training with MLflow tracking
- ✅ Interactive Streamlit dashboard
- ✅ Airflow workflow orchestration
- ✅ Evidently AI model monitoring
- ✅ FastAPI REST API + WebSocket
- ✅ Full stack Docker deployment

## 🏗️ Architecture

**Medallion Data Pipeline**:
```
News Simulator → Bronze → Silver → Gold → Model → Inference → Monitoring
      ↓            ↓        ↓        ↓      ↓        ↓           ↓
   5001 port    Raw    Features  Training  API   Dashboard   Evidently
              Storage  Engineering         8000    8501        8050
```

## 🎨 Components

### Data Layer
- **Bronze**: OANDA market data (1.7M 1-min candles) + RSS news feeds
- **Silver**: 48 engineered features (37 market + 11 news sentiment)
- **Gold**: Training-ready dataset with labels

### ML Model
- **Algorithm**: XGBoost Classifier
- **Task**: 30-minute price direction prediction (UP/DOWN)
- **Performance**: AUC 0.635, Accuracy 58.85%
- **Tracking**: MLflow experiment tracking & model registry

### Services

| Service | Port | Description | Access |
|---------|------|-------------|--------|
| **Streamlit** | 8501 | Interactive ML dashboard | http://localhost:8501 |
| **FastAPI** | 8000 | REST API + WebSocket | http://localhost:8000/docs |
| **MLflow** | 5000 | Experiment tracking | http://localhost:5000 |
| **Airflow** | 8080 | Workflow orchestration | admin/admin |
| **Evidently** | 8050 | Model monitoring | http://localhost:8050 |
| **News Simulator** | 5001 | Test data generator | http://localhost:5001 |

## 🔄 Demo Workflows

### 1. Data Ingestion Demo
```bash
# Start news simulator
cd news-simulator && python app.py &

# Generate 100 test articles
curl -X POST http://localhost:5001/api/generate \
  -d '{"count": 100, "topic": "sp500"}' \
  -H "Content-Type: application/json"

# Process to bronze layer
cp news-simulator/generated/*.json data_clean/bronze/news/raw_articles/

# Run sentiment analysis (Silver layer)
python src_clean/data_pipelines/silver/news_sentiment_processor.py \
  --input data_clean/bronze/news/raw_articles/ \
  --output data_clean/silver/news/sentiment/
```

### 2. Training Demo
```bash
# Train with MLflow tracking
python src_clean/training/xgboost_training_pipeline_mlflow.py \
  --market-features data_clean/gold/market/features/spx500_features.csv \
  --news-signals data_clean/gold/news/signals/sp500_trading_signals.csv \
  --prediction-horizon 30 \
  --experiment-name demo_experiment

# View results: http://localhost:5000
```

### 3. Inference Demo
```bash
# Start FastAPI
uvicorn src_clean.api.main:app --host 0.0.0.0 --port 8000 &

# Generate prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instrument": "SPX500_USD"}'

# Stream real-time updates (requires wscat)
wscat -c ws://localhost:8000/ws/market-stream
```

### 4. Orchestration Demo
```bash
# Start Airflow
cd airflow_mlops
docker-compose up -d airflow-webserver airflow-scheduler

# Access: http://localhost:8080 (admin/admin)
# Trigger DAGs:
#   - data_pipeline (Bronze → Silver → Gold)
#   - train_deploy_pipeline (Train & deploy model)
#   - batch_inference (Generate predictions)
```

### 5. Monitoring Demo
```bash
# Start Evidently
docker-compose up -d evidently-monitor

# Generate drift report
curl -X POST http://localhost:8050/generate

# View: http://localhost:8050/latest_report.html
```

### 6. Full Stack Demo
```bash
# Launch all services
cd docker
docker-compose -f docker-compose.full-stack.yml up -d

# Verify all services healthy
docker-compose -f docker-compose.full-stack.yml ps

# Access all UIs (see Services table above)
```

## 📊 Features

**Market Features (37)**:
- Technical: RSI, MACD, Bollinger Bands, Moving Averages, ATR, ADX
- Microstructure: Volume patterns, spread proxies, order flow
- Volatility: Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang

**News Features (11)**:
- Sentiment scores (positive/negative/neutral)
- Trading signal strength
- Article quality metrics
- Policy tone indicators

## 📁 Project Structure

```
fx-ml-pipeline/
├── README.md                    # This file
├── requirements.txt             # Python 3.11 dependencies
│
├── src_clean/                   # Production code
│   ├── api/                     # FastAPI backend
│   ├── ui/                      # Streamlit dashboards
│   ├── data_pipelines/          # Bronze → Silver → Gold
│   │   ├── bronze/              # Data collection
│   │   ├── silver/              # Feature engineering
│   │   └── gold/                # Training data prep
│   ├── training/                # XGBoost training
│   │   ├── xgboost_training_pipeline.py
│   │   └── xgboost_training_pipeline_mlflow.py
│   └── utils/                   # Shared utilities
│
├── docker/                      # Docker configurations
│   ├── Dockerfile.fastapi
│   ├── Dockerfile.streamlit
│   ├── docker-compose.yml
│   └── docker-compose.full-stack.yml
│
├── docs/                        # Documentation (local only)
│   ├── COMPLETE_DEMO_GUIDE.md   # Full system demo
│   ├── DEPLOYMENT_GUIDE.md      # Deployment instructions
│   └── IMPLEMENTATION_SUMMARY.md
│
├── data_clean/                  # Medallion data architecture
│   ├── bronze/                  # Raw data
│   ├── silver/                  # Engineered features
│   ├── gold/                    # Training-ready data
│   └── models/                  # Trained XGBoost models
│
├── feature_repo/                # Feast feature store config
├── airflow_mlops/               # Airflow DAGs & config
├── news-simulator/              # News article generator
└── archive/                     # Archived code & data
```

## 🔧 Requirements

- **Python**: 3.11+ (required)
- **Docker**: For full stack deployment
- **RAM**: 16GB minimum
- **Disk**: 50GB free space
- **OANDA Account**: Free demo account available

## 🎓 Learning Resources

- [Complete Demo Guide](docs/COMPLETE_DEMO_GUIDE.md) - Step-by-step walkthrough
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Production deployment
- [MLflow Docs](https://mlflow.org/docs/latest/)
- [Airflow Docs](https://airflow.apache.org/docs/)
- [Streamlit Docs](https://docs.streamlit.io/)

## 🐛 Troubleshooting

### Python Version
```bash
# Verify Python 3.11
python --version  # Should show 3.11.x

# Recreate venv if needed
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Port Conflicts
```bash
# Find process
lsof -i :8000  # or any port

# Kill process
kill -9 <PID>
```

### Docker Issues
```bash
# Restart containers
cd docker
docker-compose -f docker-compose.full-stack.yml restart

# View logs
docker-compose -f docker-compose.full-stack.yml logs -f <service>
```

## 📝 License

Educational and research purposes only.

---

**Version**: 2.0.0
**Python**: 3.11+
**Last Updated**: October 2025
