.PHONY: help setup clean test lint format typecheck docker-up docker-down docker-rebuild docker-logs airflow-trigger mlflow-ui airflow-ui streamlit-ui feast-apply feast-materialize validate backup archive

# Default target
.DEFAULT_GOAL := help

# Colors for output
COLOR_RESET = \033[0m
COLOR_BOLD = \033[1m
COLOR_GREEN = \033[32m
COLOR_YELLOW = \033[33m
COLOR_BLUE = \033[34m

##@ General

help: ## Display this help message
	@echo "$(COLOR_BOLD)S&P 500 ML Pipeline - Available Commands$(COLOR_RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(COLOR_BOLD)%s$(COLOR_RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Installation

setup: ## Initial project setup (venv, dependencies, Docker)
	@echo "$(COLOR_BLUE)Setting up project...$(COLOR_RESET)"
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "$(COLOR_GREEN)✓ Virtual environment created and dependencies installed$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Next steps:$(COLOR_RESET)"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Copy .env.monitoring.example to .env.monitoring and configure"
	@echo "  3. Run 'make docker-up' to start services"

install: ## Install/update Python dependencies
	@echo "$(COLOR_BLUE)Installing dependencies...$(COLOR_RESET)"
	.venv/bin/pip install -r requirements.txt
	@echo "$(COLOR_GREEN)✓ Dependencies installed$(COLOR_RESET)"

clean: ## Clean temporary files, cache, and build artifacts
	@echo "$(COLOR_BLUE)Cleaning project...$(COLOR_RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	@echo "$(COLOR_GREEN)✓ Project cleaned$(COLOR_RESET)"

##@ Code Quality

test: ## Run all tests with pytest
	@echo "$(COLOR_BLUE)Running tests...$(COLOR_RESET)"
	.venv/bin/pytest tests/ -v
	@echo "$(COLOR_GREEN)✓ Tests completed$(COLOR_RESET)"

test-coverage: ## Run tests with coverage report
	@echo "$(COLOR_BLUE)Running tests with coverage...$(COLOR_RESET)"
	.venv/bin/pytest tests/ --cov=src_clean --cov-report=html --cov-report=term
	@echo "$(COLOR_GREEN)✓ Coverage report generated in htmlcov/$(COLOR_RESET)"

lint: ## Run code linting with flake8
	@echo "$(COLOR_BLUE)Running linter...$(COLOR_RESET)"
	.venv/bin/flake8 src_clean/ tests/ --max-line-length=100 --exclude=__pycache__,.venv
	@echo "$(COLOR_GREEN)✓ Linting completed$(COLOR_RESET)"

format: ## Format code with black
	@echo "$(COLOR_BLUE)Formatting code...$(COLOR_RESET)"
	.venv/bin/black src_clean/ tests/ --line-length=100
	@echo "$(COLOR_GREEN)✓ Code formatted$(COLOR_RESET)"

typecheck: ## Run type checking with mypy
	@echo "$(COLOR_BLUE)Running type checker...$(COLOR_RESET)"
	.venv/bin/mypy src_clean/ --ignore-missing-imports
	@echo "$(COLOR_GREEN)✓ Type checking completed$(COLOR_RESET)"

##@ Docker Services

docker-up: ## Start all Docker services
	@echo "$(COLOR_BLUE)Starting Docker services...$(COLOR_RESET)"
	docker-compose up -d
	@echo "$(COLOR_GREEN)✓ Docker services started$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Service URLs:$(COLOR_RESET)"
	@echo "  Airflow:   http://localhost:8080 (admin/admin)"
	@echo "  MLflow:    http://localhost:5001"
	@echo "  FastAPI:   http://localhost:8000"
	@echo "  Streamlit: http://localhost:8501"

docker-down: ## Stop all Docker services
	@echo "$(COLOR_BLUE)Stopping Docker services...$(COLOR_RESET)"
	docker-compose down
	@echo "$(COLOR_GREEN)✓ Docker services stopped$(COLOR_RESET)"

docker-rebuild: ## Rebuild and restart Docker services
	@echo "$(COLOR_BLUE)Rebuilding Docker services...$(COLOR_RESET)"
	docker-compose down
	docker-compose up -d --build
	@echo "$(COLOR_GREEN)✓ Docker services rebuilt and restarted$(COLOR_RESET)"

docker-logs: ## View logs from all Docker services
	docker-compose logs -f

docker-clean: ## Remove all Docker containers, volumes, and images
	@echo "$(COLOR_YELLOW)Warning: This will remove all Docker containers, volumes, and images$(COLOR_RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		docker system prune -f; \
		echo "$(COLOR_GREEN)✓ Docker cleaned$(COLOR_RESET)"; \
	fi

##@ Airflow DAGs

airflow-trigger: ## Trigger main training DAG
	@echo "$(COLOR_BLUE)Triggering sp500_ml_pipeline_v4_docker DAG...$(COLOR_RESET)"
	docker-compose exec airflow-scheduler airflow dags trigger sp500_ml_pipeline_v4_docker
	@echo "$(COLOR_GREEN)✓ DAG triggered$(COLOR_RESET)"

airflow-trigger-alt: ## Trigger alternative training DAG
	@echo "$(COLOR_BLUE)Triggering sp500_ml_pipeline_v4_1_docker DAG...$(COLOR_RESET)"
	docker-compose exec airflow-scheduler airflow dags trigger sp500_ml_pipeline_v4_1_docker
	@echo "$(COLOR_GREEN)✓ DAG triggered$(COLOR_RESET)"

airflow-trigger-inference: ## Trigger online inference DAG
	@echo "$(COLOR_BLUE)Triggering sp500_online_inference_pipeline DAG...$(COLOR_RESET)"
	docker-compose exec airflow-scheduler airflow dags trigger sp500_online_inference_pipeline
	@echo "$(COLOR_GREEN)✓ DAG triggered$(COLOR_RESET)"

airflow-list: ## List all Airflow DAGs
	docker-compose exec airflow-scheduler airflow dags list

airflow-status: ## Show status of all DAGs
	docker-compose exec airflow-scheduler airflow dags list-runs

##@ UI Access

mlflow-ui: ## Open MLflow UI in browser
	@echo "$(COLOR_BLUE)Opening MLflow UI...$(COLOR_RESET)"
	open http://localhost:5001

airflow-ui: ## Open Airflow UI in browser
	@echo "$(COLOR_BLUE)Opening Airflow UI...$(COLOR_RESET)"
	@echo "Credentials: admin/admin"
	open http://localhost:8080

streamlit-ui: ## Open Streamlit dashboard in browser
	@echo "$(COLOR_BLUE)Opening Streamlit dashboard...$(COLOR_RESET)"
	open http://localhost:8501

api-docs: ## Open FastAPI documentation in browser
	@echo "$(COLOR_BLUE)Opening FastAPI docs...$(COLOR_RESET)"
	open http://localhost:8000/docs

##@ Feature Store

feast-apply: ## Apply Feast feature definitions
	@echo "$(COLOR_BLUE)Applying Feast feature definitions...$(COLOR_RESET)"
	cd feature_repo && ../.venv/bin/feast apply
	@echo "$(COLOR_GREEN)✓ Feast features applied$(COLOR_RESET)"

feast-materialize: ## Materialize features to online store
	@echo "$(COLOR_BLUE)Materializing features...$(COLOR_RESET)"
	cd feature_repo && ../.venv/bin/feast materialize-incremental $$(date -u +%Y-%m-%dT%H:%M:%S)
	@echo "$(COLOR_GREEN)✓ Features materialized$(COLOR_RESET)"

feast-ui: ## Start Feast UI
	@echo "$(COLOR_BLUE)Starting Feast UI...$(COLOR_RESET)"
	cd feature_repo && ../.venv/bin/feast ui

##@ Validation & Testing

validate: ## Run full pipeline validation
	@echo "$(COLOR_BLUE)Running pipeline validation...$(COLOR_RESET)"
	./scripts/validate_pipeline.sh
	@echo "$(COLOR_GREEN)✓ Validation completed$(COLOR_RESET)"

test-e2e: ## Run end-to-end pipeline test
	@echo "$(COLOR_BLUE)Running end-to-end test...$(COLOR_RESET)"
	./scripts/test_pipeline_e2e.sh
	@echo "$(COLOR_GREEN)✓ E2E test completed$(COLOR_RESET)"

test-drift: ## Run drift simulation test
	@echo "$(COLOR_BLUE)Running drift simulation...$(COLOR_RESET)"
	./scripts/test_drift_simulation.sh
	@echo "$(COLOR_GREEN)✓ Drift simulation completed$(COLOR_RESET)"

health-check: ## Check health of all services
	@echo "$(COLOR_BLUE)Checking service health...$(COLOR_RESET)"
	@echo "FastAPI:"
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "  ✗ FastAPI not responding"
	@echo ""
	@echo "MLflow:"
	@curl -s http://localhost:5001/health | python3 -m json.tool || echo "  ✗ MLflow not responding"
	@echo ""
	@echo "Streamlit:"
	@curl -s -o /dev/null -w "  Status: %{http_code}\n" http://localhost:8501 || echo "  ✗ Streamlit not responding"

##@ Data Management

backup: ## Create backup of data_clean directory
	@echo "$(COLOR_BLUE)Creating backup...$(COLOR_RESET)"
	tar -czf backup_data_clean_$$(date +%Y%m%d_%H%M%S).tar.gz data_clean/
	@echo "$(COLOR_GREEN)✓ Backup created$(COLOR_RESET)"

archive: ## Move old data to archive directory
	@echo "$(COLOR_BLUE)Archiving old data...$(COLOR_RESET)"
	@mkdir -p archive_$$(date +%Y_%m_%d)
	@echo "$(COLOR_YELLOW)Manual archival required - specify directories to archive$(COLOR_RESET)"

data-info: ## Show data directory sizes
	@echo "$(COLOR_BOLD)Data Directory Sizes:$(COLOR_RESET)"
	@du -sh data_clean/ 2>/dev/null || echo "  data_clean/: Not found"
	@du -sh data_clean/bronze/ 2>/dev/null || echo "  data_clean/bronze/: Not found"
	@du -sh data_clean/silver/ 2>/dev/null || echo "  data_clean/silver/: Not found"
	@du -sh data_clean/gold/ 2>/dev/null || echo "  data_clean/gold/: Not found"
	@du -sh data_clean/models/ 2>/dev/null || echo "  data_clean/models/: Not found"

##@ Development

dev: docker-up ## Start development environment (Docker + all services)
	@echo "$(COLOR_GREEN)✓ Development environment ready$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Quick Commands:$(COLOR_RESET)"
	@echo "  make airflow-trigger       - Start training pipeline"
	@echo "  make mlflow-ui             - View experiments"
	@echo "  make docker-logs           - View service logs"
	@echo "  make health-check          - Check all services"

stop: docker-down ## Stop development environment

restart: docker-rebuild ## Restart all services with rebuild

watch-logs: ## Watch logs for specific service (usage: make watch-logs SERVICE=airflow-scheduler)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(COLOR_YELLOW)Usage: make watch-logs SERVICE=<service-name>$(COLOR_RESET)"; \
		echo "Available services:"; \
		docker-compose ps --services; \
	else \
		docker-compose logs -f $(SERVICE); \
	fi

shell: ## Open shell in specific service (usage: make shell SERVICE=airflow-scheduler)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(COLOR_YELLOW)Usage: make shell SERVICE=<service-name>$(COLOR_RESET)"; \
		echo "Available services:"; \
		docker-compose ps --services; \
	else \
		docker-compose exec $(SERVICE) /bin/bash; \
	fi

##@ Monitoring

monitor-drift: ## View latest drift report
	@echo "$(COLOR_BLUE)Latest drift reports:$(COLOR_RESET)"
	@ls -lt data_clean/monitoring/*.html | head -5

show-predictions: ## Show recent predictions
	@echo "$(COLOR_BLUE)Recent predictions:$(COLOR_RESET)"
	@tail -20 data_clean/predictions/prediction_log.jsonl

show-models: ## List trained models
	@echo "$(COLOR_BOLD)Trained Models:$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_GREEN)XGBoost Models:$(COLOR_RESET)"
	@ls -1 data_clean/models/xgboost/ 2>/dev/null || echo "  None found"
	@echo ""
	@echo "$(COLOR_GREEN)LightGBM Models:$(COLOR_RESET)"
	@ls -1 data_clean/models/lightgbm/ 2>/dev/null || echo "  None found"
	@echo ""
	@echo "$(COLOR_GREEN)AR Models:$(COLOR_RESET)"
	@ls -1 data_clean/models/ar/ 2>/dev/null || echo "  None found"
	@echo ""
	@echo "$(COLOR_GREEN)Production Models:$(COLOR_RESET)"
	@ls -1 data_clean/models/production/ 2>/dev/null || echo "  None found"

##@ Documentation

docs: ## Generate documentation (placeholder for future Sphinx integration)
	@echo "$(COLOR_YELLOW)Documentation generation not yet implemented$(COLOR_RESET)"
	@echo "For now, see: docs/Technical_Report_MLOps.md"

serve-docs: ## Serve documentation locally (placeholder)
	@echo "$(COLOR_YELLOW)Documentation server not yet implemented$(COLOR_RESET)"
	@echo "For now, open: docs/Technical_Report_MLOps.md"
