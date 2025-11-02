# Environment Setup Guide

## Quick Setup for New Users

This project uses dynamic path resolution to work on any machine. Follow these steps:

### 1. Create your environment file

Copy the example environment file and update it with your local paths:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and set your project root path
# Windows example: FX_ML_PIPELINE_ROOT=c:/Users/YourName/path/to/fx-ml-pipeline
# Mac/Linux example: FX_ML_PIPELINE_ROOT=/Users/YourName/path/to/fx-ml-pipeline
```

**Important:** Use forward slashes (`/`) even on Windows for Docker compatibility.

### 2. Automatic Path Detection (Alternative)

If you don't want to set up the `.env` file, the system will automatically detect the project root from the DAG file location. This works out of the box, but setting the environment variable is more explicit and recommended for production.

### 3. Start the services

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f airflow-scheduler
```

### 4. Verify the setup

The Airflow DAG will automatically use the correct paths based on:
1. First priority: `FX_ML_PIPELINE_ROOT` environment variable from `.env`
2. Fallback: Auto-detection from DAG file location

You should see log messages in Airflow indicating the detected project root path.

## Troubleshooting

### Path Issues

If you see errors like "bind source path does not exist", check:

1. **Is `.env` file present?**
   ```bash
   cat .env
   ```

2. **Is the path correct in `.env`?**
   - Should be the absolute path to your project root
   - Must use forward slashes (`/`) even on Windows
   - No trailing slash

3. **Did you restart docker-compose after creating .env?**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Windows-specific Notes

- Always use forward slashes in paths: `c:/Users/...` not `c:\Users\...`
- Docker Desktop must be running
- Ensure your project is in a shared drive (Docker Desktop settings)

### Mac/Linux-specific Notes

- Paths should start with `/Users/` or `/home/`
- Ensure Docker has permission to access the directory

## How It Works

The DAG file `airflow_mlops/dags/sp500_ml_pipeline_v4_docker.py` uses this logic:

```python
# 1. Try environment variable first
PROJECT_ROOT = os.getenv('FX_ML_PIPELINE_ROOT')

# 2. Fall back to auto-detection
if not PROJECT_ROOT:
    dag_file_path = Path(__file__).resolve()
    PROJECT_ROOT = str(dag_file_path.parent.parent.parent)

# 3. Normalize for Docker (forward slashes)
PROJECT_ROOT = PROJECT_ROOT.replace('\\', '/')
```

This ensures the pipeline works on Windows, Mac, and Linux without hardcoded paths.
