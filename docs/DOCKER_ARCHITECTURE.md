# Docker Architecture Design

> **Related**: [MLOPS_PLAN.md](MLOPS_PLAN.md)

---

## 🐳 Total Docker Images: 11

### **Image Breakdown**

| # | Image Name | Base Image | Purpose | Build Type | GPU Required |
|---|------------|------------|---------|------------|--------------|
| 1 | `fx-base` | `python:3.10-slim` | Shared dependencies | Custom | No |
| 2 | `fx-airflow-scheduler` | `apache/airflow:2.7.0` | DAG scheduling | Custom | No |
| 3 | `fx-airflow-webserver` | `apache/airflow:2.7.0` | Airflow UI | Custom | No |
| 4 | `fx-airflow-worker` | `apache/airflow:2.7.0` | Task execution | Custom | No |
| 5 | `fx-fastapi` | `fx-base` | Inference API | Custom | No |
| 6 | `fx-fingpt` | `nvidia/cuda:11.8.0-runtime` | FinGPT service | Custom | Yes |
| 7 | `fx-mlflow` | `python:3.10-slim` | Model registry | Custom | No |
| 8 | `postgres:14` | Official | Airflow metadata | Official | No |
| 9 | `redis:7` | Official | Feast online store | Official | No |
| 10 | `prom/prometheus:latest` | Official | Metrics collection | Official | No |
| 11 | `grafana/grafana:latest` | Official | Dashboards | Official | No |

**Custom Images**: 7 (need Dockerfiles)
**Official Images**: 4 (pulled from Docker Hub)

---

## 📦 Detailed Image Specifications

### **1. fx-base (Shared Base Image)**

**Purpose**: Common dependencies for all Python services to reduce duplication

**Dockerfile.base**:
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install common Python packages
COPY docker/requirements.base.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.base.txt

# Create non-root user
RUN useradd -m -u 1000 fxuser && \
    mkdir -p /app /data && \
    chown -R fxuser:fxuser /app /data

WORKDIR /app
USER fxuser
```

**requirements.base.txt**:
```
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
pydantic==2.4.0
pyyaml==6.0.1
python-dotenv==1.0.0
requests==2.31.0
```

**Size Estimate**: ~800 MB

---

### **2. fx-airflow-scheduler**

**Purpose**: Airflow scheduler for DAG execution

**Dockerfile.airflow-scheduler**:
```dockerfile
FROM apache/airflow:2.7.0-python3.10

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python dependencies
COPY docker/requirements.airflow.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.airflow.txt

# Install project dependencies
COPY pyproject.toml /tmp/
RUN pip install --no-cache-dir feast mlflow xgboost

# Set environment variables
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth \
    PYTHONPATH=/opt/airflow/src:$PYTHONPATH

# Copy Airflow DAGs and plugins
COPY --chown=airflow:root airflow/dags /opt/airflow/dags
COPY --chown=airflow:root airflow/plugins /opt/airflow/plugins

# Initialize Airflow DB and create admin user (done in entrypoint)
COPY docker/airflow-init.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["scheduler"]
```

**requirements.airflow.txt**:
```
apache-airflow-providers-postgres==5.7.1
apache-airflow-providers-redis==3.3.0
feast==0.34.1
mlflow==2.9.2
xgboost==2.0.3
prometheus-client==0.19.0
```

**Size Estimate**: ~2.5 GB

---

### **3. fx-airflow-webserver**

**Purpose**: Airflow web UI

**Dockerfile.airflow-webserver**:
```dockerfile
FROM apache/airflow:2.7.0-python3.10

USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

USER airflow

COPY docker/requirements.airflow.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.airflow.txt

ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True

COPY --chown=airflow:root airflow/dags /opt/airflow/dags
COPY --chown=airflow:root airflow/plugins /opt/airflow/plugins

COPY docker/airflow-init.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
CMD ["webserver"]
```

**Size Estimate**: ~2.5 GB

---

### **4. fx-airflow-worker**

**Purpose**: Execute Airflow tasks (data collection, feature engineering, training)

**Dockerfile.airflow-worker**:
```dockerfile
FROM apache/airflow:2.7.0-python3.10

USER root
RUN apt-get update && apt-get install -y \
    git gcc g++ \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY docker/requirements.airflow.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.airflow.txt

# Install project source code as package
COPY --chown=airflow:root pyproject.toml /opt/airflow/
COPY --chown=airflow:root src/ /opt/airflow/src/
RUN pip install --no-cache-dir -e /opt/airflow

ENV PYTHONPATH=/opt/airflow/src:$PYTHONPATH

COPY docker/airflow-init.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["worker"]
```

**Key Difference from Scheduler/Webserver**:
- Includes full project source code (`src/`)
- Executes Python scripts (data collection, feature engineering, training)

**Size Estimate**: ~2.8 GB

---

### **5. fx-fastapi**

**Purpose**: REST API for real-time inference

**Dockerfile.fastapi**:
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY fastapi_service/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
COPY fastapi_service/ /app/

# Copy Feast feature repo
COPY feast_repo/ /app/feast_repo/

# Create non-root user
RUN useradd -m -u 1000 fxuser && \
    chown -R fxuser:fxuser /app
USER fxuser

EXPOSE 8000

# Production server with multiple workers
CMD ["gunicorn", "app.main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

**requirements.txt** (fastapi_service/):
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
gunicorn==21.2.0
feast==0.34.1
mlflow==2.9.2
redis==5.0.1
prometheus-client==0.19.0
pydantic==2.5.0
python-multipart==0.0.6
```

**Size Estimate**: ~1.2 GB

---

### **6. fx-fingpt (GPU-Accelerated)**

**Purpose**: FinGPT sentiment analysis service

**Dockerfile.fingpt**:
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install Transformers and FinGPT dependencies
COPY docker/requirements.fingpt.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.fingpt.txt

# Copy FinGPT processor
COPY src/fingpt_processor.py /app/
COPY src/__init__.py /app/

# Download FinGPT model at build time (optional - can be runtime)
# RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoTokenizer.from_pretrained('FinGPT/fingpt-sentiment_llama2-7b_lora'); \
#     AutoModelForCausalLM.from_pretrained('FinGPT/fingpt-sentiment_llama2-7b_lora')"

# Create non-root user
RUN useradd -m -u 1000 fxuser && \
    chown -R fxuser:fxuser /app
USER fxuser

EXPOSE 8001

# Run FinGPT as API service
CMD ["python3", "fingpt_processor.py", "--serve", "--port", "8001", "--device", "cuda"]
```

**requirements.fingpt.txt**:
```
transformers==4.36.0
accelerate==0.25.0
bitsandbytes==0.41.3
sentencepiece==0.1.99
fastapi==0.104.1
uvicorn==0.24.0
```

**Size Estimate**: ~8 GB (includes CUDA runtime + PyTorch)

**GPU Requirements**:
- NVIDIA GPU with CUDA 11.8+ support
- Minimum 8GB VRAM for LLaMA2-7B with 8-bit quantization
- Docker with NVIDIA Container Toolkit (`nvidia-docker2`)

---

### **7. fx-mlflow**

**Purpose**: MLflow tracking server and model registry

**Dockerfile.mlflow**:
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.29.0

# Create directories
RUN mkdir -p /mlflow/artifacts /mlflow/db && \
    useradd -m -u 1000 mlflowuser && \
    chown -R mlflowuser:mlflowuser /mlflow

USER mlflowuser

EXPOSE 5000

# Start MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
```

**Size Estimate**: ~600 MB

---

### **8-11. Official Images (No Dockerfile Needed)**

#### **8. postgres:14**
- **Purpose**: Airflow metadata database + MLflow backend
- **Configuration**: Via environment variables in docker-compose.yml
- **Size**: ~350 MB

#### **9. redis:7**
- **Purpose**: Feast online feature store
- **Configuration**: Via docker-compose.yml
- **Size**: ~120 MB

#### **10. prom/prometheus:latest**
- **Purpose**: Metrics collection
- **Configuration**: Via prometheus.yml config file
- **Size**: ~250 MB

#### **11. grafana/grafana:latest**
- **Purpose**: Dashboards and visualization
- **Configuration**: Via provisioning configs
- **Size**: ~400 MB

---

## 📂 Project Directory Structure for Docker

```
fx-ml-pipeline/
├── docker/
│   ├── Dockerfile.base                    # Image 1: fx-base
│   ├── Dockerfile.airflow-scheduler       # Image 2
│   ├── Dockerfile.airflow-webserver       # Image 3
│   ├── Dockerfile.airflow-worker          # Image 4
│   ├── Dockerfile.fastapi                 # Image 5
│   ├── Dockerfile.fingpt                  # Image 6
│   ├── Dockerfile.mlflow                  # Image 7
│   ├── docker-compose.yml                 # Orchestration
│   ├── docker-compose.gpu.yml             # GPU overrides for FinGPT
│   ├── .env.example                       # Environment variables
│   ├── requirements.base.txt              # Base dependencies
│   ├── requirements.airflow.txt           # Airflow dependencies
│   ├── requirements.fingpt.txt            # FinGPT dependencies
│   └── airflow-init.sh                    # Airflow initialization script
├── airflow/
│   ├── dags/                              # Copied to Airflow images
│   └── plugins/                           # Custom Airflow operators
├── fastapi_service/                       # Copied to fx-fastapi image
│   ├── app/
│   ├── tests/
│   └── requirements.txt
├── feast_repo/                            # Copied to fx-fastapi & fx-airflow-worker
│   ├── feature_store.yaml
│   └── features/
├── src/                                   # Copied to fx-airflow-worker
│   ├── build_market_features.py
│   ├── fingpt_processor.py
│   └── ...
├── monitoring/
│   ├── prometheus.yml                     # Mounted to prometheus container
│   └── grafana/
│       ├── dashboards/                    # Mounted to grafana container
│       └── datasources.yml
└── data/                                  # Shared volume across containers
    ├── bronze/
    ├── market/
    ├── news/
    └── ...
```

---

## 🔧 Docker Compose Configuration

### **docker-compose.yml** (Main file)

```yaml
version: '3.8'

services:
  # ============= Data Layer =============
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
      # MLflow database
      POSTGRES_MULTIPLE_DATABASES: airflow,mlflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./docker/init-databases.sh:/docker-entrypoint-initdb.d/init-databases.sh
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U airflow"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ============= Airflow Components =============
  airflow-init:
    build:
      context: .
      dockerfile: docker/Dockerfile.airflow-scheduler
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db init
        airflow users create \
          --username admin \
          --password admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@fx-ml.local
    environment:
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    depends_on:
      postgres:
        condition: service_healthy

  airflow-scheduler:
    build:
      context: .
      dockerfile: docker/Dockerfile.airflow-scheduler
    depends_on:
      airflow-init:
        condition: service_completed_successfully
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__FERNET_KEY: ${AIRFLOW_FERNET_KEY}
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
      - ./data:/opt/airflow/data
      - ./src:/opt/airflow/src
      - ./feast_repo:/opt/airflow/feast_repo
      - airflow-logs:/opt/airflow/logs
    restart: unless-stopped

  airflow-webserver:
    build:
      context: .
      dockerfile: docker/Dockerfile.airflow-webserver
    depends_on:
      airflow-scheduler:
        condition: service_started
    environment:
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
    ports:
      - "8080:8080"
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
      - airflow-logs:/opt/airflow/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  airflow-worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.airflow-worker
    depends_on:
      airflow-scheduler:
        condition: service_started
    environment:
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      OANDA_API_TOKEN: ${OANDA_API_TOKEN}
      OANDA_ACCOUNT_ID: ${OANDA_ACCOUNT_ID}
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./data:/opt/airflow/data
      - ./src:/opt/airflow/src
      - ./feast_repo:/opt/airflow/feast_repo
      - ./configs:/opt/airflow/configs
      - airflow-logs:/opt/airflow/logs
    restart: unless-stopped

  # ============= ML Services =============
  mlflow:
    build:
      context: .
      dockerfile: docker/Dockerfile.mlflow
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow@postgres:5432/mlflow
      MLFLOW_ARTIFACT_ROOT: /mlflow/artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  fastapi:
    build:
      context: .
      dockerfile: docker/Dockerfile.fastapi
    depends_on:
      redis:
        condition: service_healthy
      mlflow:
        condition: service_healthy
    environment:
      FEAST_REDIS_HOST: redis
      FEAST_REDIS_PORT: 6379
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports:
      - "8000:8000"
    volumes:
      - ./fastapi_service:/app
      - ./feast_repo:/app/feast_repo
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ============= Monitoring =============
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
      - grafana-data:/var/lib/grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: false
      GF_SERVER_ROOT_URL: http://localhost:3000
    restart: unless-stopped

  redis-exporter:
    image: oliver006/redis_exporter:latest
    depends_on:
      - redis
    ports:
      - "9121:9121"
    environment:
      REDIS_ADDR: redis:6379
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  mlflow-artifacts:
  prometheus-data:
  grafana-data:
  airflow-logs:

networks:
  default:
    name: fx-ml-network
```

### **docker-compose.gpu.yml** (GPU override for FinGPT)

```yaml
version: '3.8'

services:
  fingpt:
    build:
      context: .
      dockerfile: docker/Dockerfile.fingpt
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      CUDA_VISIBLE_DEVICES: 0
      TRANSFORMERS_CACHE: /app/models
    ports:
      - "8001:8001"
    volumes:
      - ./src:/app
      - ./models:/app/models  # Cache FinGPT model
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 60s
      timeout: 10s
      retries: 3
```

**Usage**:
```bash
# CPU-only (no FinGPT)
docker-compose up -d

# With GPU (FinGPT enabled)
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## 🚀 Build & Deployment Commands

### **Initial Setup**

```bash
# 1. Clone repository
git clone <repo-url>
cd fx-ml-pipeline

# 2. Copy environment file
cp docker/.env.example docker/.env
# Edit docker/.env with your OANDA_API_TOKEN, etc.

# 3. Generate Airflow Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Add to docker/.env: AIRFLOW_FERNET_KEY=<generated_key>
```

### **Build All Custom Images**

```bash
# Build all images (from project root)
docker-compose -f docker/docker-compose.yml build

# Build specific image
docker-compose -f docker/docker-compose.yml build fastapi

# Build with no cache (force rebuild)
docker-compose -f docker/docker-compose.yml build --no-cache
```

### **Run Stack**

```bash
# Start all services (CPU-only)
docker-compose -f docker/docker-compose.yml up -d

# Start with FinGPT (GPU required)
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f fastapi

# Stop all services
docker-compose -f docker/docker-compose.yml down

# Stop and remove volumes (clean slate)
docker-compose -f docker/docker-compose.yml down -v
```

### **Development Workflow**

```bash
# Rebuild after code changes (preserves data volumes)
docker-compose -f docker/docker-compose.yml up -d --build fastapi

# Execute commands in running container
docker-compose -f docker/docker-compose.yml exec airflow-worker bash
docker-compose -f docker/docker-compose.yml exec fastapi python -c "from feast import FeatureStore; print(FeatureStore())"

# View resource usage
docker stats
```

---

## 📊 Image Size Summary

| Image | Size (Approx) | Notes |
|-------|---------------|-------|
| fx-base | 800 MB | Shared base (not deployed directly) |
| fx-airflow-scheduler | 2.5 GB | Airflow + Python deps |
| fx-airflow-webserver | 2.5 GB | Same as scheduler |
| fx-airflow-worker | 2.8 GB | Scheduler + project code |
| fx-fastapi | 1.2 GB | Lightweight inference |
| fx-fingpt | 8 GB | CUDA runtime + PyTorch + Transformers |
| fx-mlflow | 600 MB | MLflow server |
| postgres:14 | 350 MB | Official image |
| redis:7 | 120 MB | Official image |
| prom/prometheus | 250 MB | Official image |
| grafana/grafana | 400 MB | Official image |
| **Total** | **~19.5 GB** | **Without FinGPT: ~11.5 GB** |

**Storage Requirements**:
- **Images**: ~20 GB
- **Volumes** (data/models/logs): ~50-100 GB (depends on data collection duration)
- **Total Disk**: ~70-120 GB recommended

---

## 🔐 Security Considerations

### **Image Security**

1. **Non-root Users**: All custom images run as non-root user (fxuser, mlflowuser, airflow)
2. **Minimal Base Images**: Use `-slim` variants to reduce attack surface
3. **No Secrets in Images**: All credentials via environment variables or mounted secrets
4. **Regular Updates**: Pin versions but update base images regularly

### **Network Security**

```yaml
# Add to docker-compose.yml for production
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

services:
  fastapi:
    networks:
      - frontend  # Public-facing
      - backend   # Access to Redis/MLflow

  redis:
    networks:
      - backend  # Internal only
```

### **Secrets Management**

```bash
# Use Docker secrets instead of environment variables
docker secret create oanda_token ./secrets/oanda_token.txt
docker secret create mlflow_db_password ./secrets/mlflow_db_password.txt
```

---

## 🧪 Testing Docker Images

### **Unit Tests per Image**

```bash
# Test fx-fastapi
docker-compose -f docker/docker-compose.yml run --rm fastapi pytest tests/

# Test Airflow DAG syntax
docker-compose -f docker/docker-compose.yml run --rm airflow-worker \
  python -m pytest airflow/dags/tests/

# Test FinGPT service
docker-compose -f docker/docker-compose.yml run --rm fingpt \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Integration Tests**

```bash
# End-to-end test script
docker-compose -f docker/docker-compose.yml exec airflow-worker \
  python scripts/test_e2e_pipeline.py
```

---

## 📝 Next Steps

1. **Create Dockerfiles**: Write all 7 custom Dockerfiles
2. **Build Base Image**: Start with fx-base for dependency sharing
3. **Test Individual Images**: Ensure each builds successfully
4. **Configure docker-compose.yml**: Wire up all services
5. **Test GPU Setup**: Verify FinGPT with NVIDIA runtime
6. **Document Deployment**: Add operational runbook

---

**Total Images**: 11 (7 custom + 4 official)
**Build Time**: ~30-45 minutes for all images (first build)
**Deployment Time**: ~5 minutes (after images built)
