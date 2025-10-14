# Airflow MLOps

# Docker setup
 - Step 1: Up airflow-init to init airflow and create admin user
```bash
    # Init airflow
    cd airflow_mlops 
    docker-compose build
    docker-compose up airflow-init
```

 - Step 2: Up postgres container to store all airflow data
```bash
    # Run in a separate terminal
    cd airflow_mlops 
    docker-compose up postgres-airflow
```
 - Step 3: Up web, scheduler and dag processor containers
```bash
    docker-compose up airflow-web airflow-scheduler airflow-dag-processor
```

# Access Airflow UI
 - Open browser: http://localhost:8080/

# Model Inference 

Start model containers
```bash
docker compose up -d model-blue model-green
```

Check health:
```bash
curl -s http://localhost:8001/health
curl -s http://localhost:8002/health
```
Test Predictions:
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"feat_mean": 2.5}' \
  http://localhost:8001/predict

curl -s -H "Content-Type: application/json" \
  -d '{"feat_mean": 0.5}' \
  http://localhost:8002/predict
```

# Nginx Load balancer 

Start nginx-gateway
```bash
docker compose up -d nginx-gateway
```

Test nginx routing
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"feat_mean": 1.2}' \
  http://localhost:8088/predict-blue

curl -s -H "Content-Type: application/json" \
  -d '{"feat_mean": 3.7}' \
  http://localhost:8088/predict-green
```

Test load-balancing upstream:

```bash
curl -s -H "Content-Type: application/json" \
  -d '{"feat_mean": 2.5}' \
  http://localhost:8088/predict
```