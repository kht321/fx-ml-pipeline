# Docker Setup Guide

Complete guide for running the S&P 500 ML Pipeline using Docker.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Available Services](#available-services)
- [Common Use Cases](#common-use-cases)
- [Docker Commands](#docker-commands)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Docker** (v20.10+)
   ```bash
   # Check Docker version
   docker --version

   # Install Docker Desktop (Mac/Windows)
   # Visit: https://www.docker.com/products/docker-desktop
   ```

2. **Docker Compose** (v2.0+)
   ```bash
   # Check Docker Compose version
   docker-compose --version

   # Usually included with Docker Desktop
   ```

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 4 GB | 8 GB+ |
| **Disk Space** | 10 GB | 20 GB+ |
| **CPU** | 2 cores | 4+ cores |
| **GPU** | Not required | Optional (for FinGPT) |

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/[your-username]/fx-ml-pipeline.git
cd fx-ml-pipeline
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your OANDA credentials
nano .env  # or use your preferred editor
```

**Minimum required**:
```env
OANDA_TOKEN=your_token_here
OANDA_ACCOUNT_ID=your_account_id_here
```

Get free OANDA credentials at: https://developer.oanda.com/

### 3. Build Docker Images

```bash
# Build all images
docker-compose build

# Or build specific image
docker-compose build dev
```

**Build time**: 5-10 minutes (first time)

### 4. Start Services

```bash
# Start development environment
docker-compose up dev

# Or run in background (detached mode)
docker-compose up -d dev
```

---

## Available Services

The docker-compose configuration provides multiple specialized services:

### 1. **Development Environment** (`dev`)

Interactive shell with all development tools.

```bash
docker-compose run --rm dev
```

**Use cases**:
- Interactive Python shell
- Running scripts manually
- Debugging
- Testing

**Inside container**:
```bash
# Download data
python src/download_sp500_historical.py --years 5 --granularity M1

# Run pipeline
python run_sp500_pipeline.py --skip-labels

# Start Python REPL
python
```

### 2. **Jupyter Lab** (`jupyter`)

Interactive notebooks for analysis and exploration.

```bash
docker-compose up -d jupyter
```

**Access**: http://localhost:8888

**Features**:
- JupyterLab interface
- Pre-installed data science libraries
- Access to all pipeline code
- Persistent notebooks

### 3. **Data Downloader** (`downloader`)

Downloads S&P 500 historical data from OANDA.

```bash
# Download 5 years of data
docker-compose run --rm downloader

# Custom parameters
docker-compose run --rm downloader \
  python src/download_sp500_historical.py --years 10 --granularity M1
```

**Output**: `data/bronze/prices/spx500_usd_m1_5years.ndjson`

### 4. **Pipeline Runner** (`pipeline`)

Runs the complete Bronze → Silver → Gold pipeline.

```bash
docker-compose run --rm pipeline
```

**Processes**:
- Technical features
- Microstructure signals
- Volatility estimators
- Feature merging

**Output**: `data/sp500/gold/training/sp500_features.csv`

### 5. **News Scraper** (`news-scraper`)

Scrapes financial news from free RSS feeds.

```bash
docker-compose run --rm news-scraper
```

**Sources**: Yahoo Finance, CNBC, MarketWatch, Seeking Alpha

**Output**: `data/news/bronze/raw_articles/*.json`

### 6. **News Processor** (`news-processor`)

Processes news articles into trading signals.

```bash
docker-compose run --rm news-processor
```

**Output**: `data/news/gold/news_signals/sp500_trading_signals.csv`

### 7. **Model Trainer** (`trainer`)

Trains XGBoost models on combined features.

```bash
docker-compose run --rm trainer
```

**Output**: `models/*.pkl`

### 8. **API Server** (`api`)

Model serving API (FastAPI-based).

```bash
docker-compose up -d api
```

**Access**: http://localhost:8000

**Endpoints**:
- `GET /health` - Health check
- `POST /predict` - Make predictions

### 9. **Supporting Services**

**Redis** (Feature Store):
```bash
docker-compose up -d redis
```
Access: `localhost:6379`

**PostgreSQL** (Metadata):
```bash
docker-compose up -d postgres
```
Access: `localhost:5432`

---

## Common Use Cases

### Use Case 1: First Time Setup

Complete workflow from scratch:

```bash
# 1. Build images
docker-compose build

# 2. Download data (5 years)
docker-compose run --rm downloader

# 3. Process market data
docker-compose run --rm pipeline

# 4. Collect news data
docker-compose run --rm news-scraper
docker-compose run --rm news-processor

# 5. Train models
docker-compose run --rm trainer
```

### Use Case 2: Daily Data Collection

Run as a scheduled job:

```bash
# Download latest data
docker-compose run --rm downloader \
  python src/download_sp500_historical.py --years 1 --granularity M1

# Scrape latest news
docker-compose run --rm news-scraper

# Update features
docker-compose run --rm pipeline
docker-compose run --rm news-processor
```

### Use Case 3: Interactive Development

```bash
# Start Jupyter Lab
docker-compose up -d jupyter

# Open browser to http://localhost:8888
# Create notebooks in /notebooks directory

# When done:
docker-compose down jupyter
```

### Use Case 4: Model Experimentation

```bash
# Start dev environment
docker-compose run --rm dev /bin/bash

# Inside container:
cd /app
python src/train_combined_model.py --help
python src/train_combined_model.py \
  --market-features data/sp500/gold/training/sp500_features.csv \
  --news-features data/news/gold/news_signals/sp500_trading_signals.csv
```

### Use Case 5: Production Deployment

```bash
# Start all services
docker-compose up -d api redis postgres

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

---

## Docker Commands

### Building

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build dev

# Build without cache (clean rebuild)
docker-compose build --no-cache

# Build with specific target
docker build --target production -t sp500-pipeline:prod .
```

### Running

```bash
# Run service (removes container after exit)
docker-compose run --rm SERVICE_NAME

# Run with custom command
docker-compose run --rm dev python --version

# Start service in background
docker-compose up -d SERVICE_NAME

# Start all services
docker-compose up -d

# View running containers
docker-compose ps
```

### Managing

```bash
# Stop services
docker-compose stop

# Stop specific service
docker-compose stop jupyter

# Remove stopped containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Restart service
docker-compose restart SERVICE_NAME
```

### Logs & Debugging

```bash
# View logs
docker-compose logs SERVICE_NAME

# Follow logs (tail -f)
docker-compose logs -f SERVICE_NAME

# View logs for all services
docker-compose logs -f

# Execute command in running container
docker-compose exec SERVICE_NAME bash

# Example: Access dev container shell
docker-compose exec dev /bin/bash
```

### Data & Volumes

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect fx-ml-pipeline_postgres-data

# Remove specific volume
docker volume rm fx-ml-pipeline_redis-data

# Backup volume
docker run --rm \
  -v fx-ml-pipeline_postgres-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/postgres-backup.tar.gz /data
```

---

## Troubleshooting

### Issue 1: Build Fails

**Problem**: Docker build fails with dependency errors

**Solution**:
```bash
# Clean rebuild
docker-compose build --no-cache

# Check Docker resources (increase if needed)
# Docker Desktop → Settings → Resources
```

### Issue 2: Permission Errors

**Problem**: Cannot write to mounted volumes

**Solution**:
```bash
# Fix permissions on host
chmod -R 755 data/ models/ outputs/

# Or run with host user ID
docker-compose run --rm --user $(id -u):$(id -g) dev
```

### Issue 3: Container Exits Immediately

**Problem**: Service container stops right after starting

**Solution**:
```bash
# Check logs
docker-compose logs SERVICE_NAME

# Run with interactive shell
docker-compose run --rm SERVICE_NAME /bin/bash

# Check environment variables
docker-compose config
```

### Issue 4: Port Already in Use

**Problem**: "Port 8888 is already allocated"

**Solution**:
```bash
# Stop conflicting service
lsof -ti:8888 | xargs kill -9

# Or change port in docker-compose.yml
ports:
  - "8889:8888"  # Use different host port
```

### Issue 5: Out of Memory

**Problem**: Container killed due to OOM

**Solution**:
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory

# Or limit container memory
docker-compose run --rm --memory="4g" trainer
```

### Issue 6: Data Not Persisting

**Problem**: Data disappears after container restart

**Solution**:
```bash
# Check volume mounts in docker-compose.yml
volumes:
  - ./data:/app/data  # Host path : Container path

# Verify volume exists
docker volume ls
```

### Issue 7: Network Issues

**Problem**: Cannot connect to OANDA API

**Solution**:
```bash
# Check environment variables
docker-compose run --rm dev env | grep OANDA

# Test API connectivity
docker-compose run --rm dev \
  python -c "import requests; print(requests.get('https://api-fxpractice.oanda.com/v3/accounts').status_code)"

# Check network
docker network ls
docker network inspect fx-ml-pipeline_sp500-network
```

---

## Best Practices

### 1. Environment Variables

Always use `.env` file, never commit credentials:

```bash
# Good
OANDA_TOKEN=${OANDA_TOKEN}

# Bad - hardcoded
OANDA_TOKEN=abc123xyz
```

### 2. Volume Mounts

Use named volumes for persistent data:

```yaml
# Good - named volume
volumes:
  - postgres-data:/var/lib/postgresql/data

# Acceptable - bind mount for development
volumes:
  - ./data:/app/data
```

### 3. Container Cleanup

Remove stopped containers regularly:

```bash
# Remove all stopped containers
docker-compose down

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune
```

### 4. Resource Limits

Set memory/CPU limits in production:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### 5. Health Checks

Monitor service health:

```bash
# Check health status
docker-compose ps

# View health check logs
docker inspect --format='{{json .State.Health}}' CONTAINER_ID | python -m json.tool
```

---

## Advanced Usage

### Multi-Stage Builds

Use different stages for different purposes:

```bash
# Development image (with dev tools)
docker build --target development -t sp500:dev .

# Production image (minimal)
docker build --target production -t sp500:prod .

# Jupyter image
docker build --target jupyter -t sp500:jupyter .
```

### Custom Networks

Create isolated networks:

```bash
# Create custom network
docker network create sp500-prod

# Run container on custom network
docker run --network sp500-prod sp500:prod
```

### Scheduled Execution

Use with cron for automation:

```bash
# Edit crontab
crontab -e

# Add daily data collection (6 AM)
0 6 * * * cd /path/to/fx-ml-pipeline && docker-compose run --rm downloader >> /var/log/sp500-download.log 2>&1

# Add hourly news scraping
0 * * * * cd /path/to/fx-ml-pipeline && docker-compose run --rm news-scraper >> /var/log/sp500-news.log 2>&1
```

---

## Performance Tips

1. **Build Cache**: Keep requirements.txt stable to leverage Docker layer caching
2. **Image Size**: Use alpine or slim base images when possible
3. **Volume Performance**: Use named volumes instead of bind mounts in production
4. **Network**: Use `--network=host` for better network performance (Linux only)
5. **Cleanup**: Regularly prune unused images, containers, and volumes

---

## Security Considerations

1. **Never commit** `.env` file or credentials
2. **Use secrets** management for production (Docker secrets, HashiCorp Vault)
3. **Run as non-root** user (already configured in production image)
4. **Update base images** regularly for security patches
5. **Scan images** for vulnerabilities: `docker scan sp500-pipeline:prod`

---

## Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **Docker Compose**: https://docs.docker.com/compose/
- **Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **Security**: https://docs.docker.com/engine/security/

---

## Getting Help

**Issue**: Container not starting?
```bash
docker-compose logs SERVICE_NAME
```

**Issue**: Data not accessible?
```bash
docker-compose run --rm dev ls -la /app/data
```

**Issue**: Need to debug?
```bash
docker-compose run --rm dev /bin/bash
cd /app
ls -la
python --version
```

---

## Summary of Commands

```bash
# Quick Start
docker-compose build
docker-compose run --rm downloader
docker-compose run --rm pipeline
docker-compose up -d jupyter

# Daily Operations
docker-compose run --rm news-scraper
docker-compose run --rm trainer

# Cleanup
docker-compose down
docker system prune -a
```

---

**Last Updated**: 2025-10-13

**Repository**: https://github.com/[your-username]/fx-ml-pipeline
