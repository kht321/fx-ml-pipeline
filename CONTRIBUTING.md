# Contributing to S&P 500 ML Pipeline

Thank you for your interest in contributing to this MLOps project! This document provides guidelines for contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 16GB+ RAM recommended
- macOS, Linux, or WSL2 on Windows

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/fx-ml-pipeline.git
   cd fx-ml-pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   cp .env.monitoring.example .env.monitoring
   # Edit .env and .env.monitoring with your configuration
   ```

5. **Start Docker services**
   ```bash
   docker-compose up -d
   ```

6. **Verify setup**
   ```bash
   make test  # Or: python -m pytest tests/
   ```

---

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

- **Description**: Clear and concise description of the bug
- **Steps to Reproduce**: Detailed steps to reproduce the behavior
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: OS, Python version, Docker version
- **Logs**: Relevant logs from `logs/` directory or Docker containers

**Template:**
```markdown
**Bug Description:**
[Description here]

**Steps to Reproduce:**
1. Start services with `docker-compose up -d`
2. Trigger DAG `sp500_ml_pipeline_v4_docker`
3. Observe error in Airflow UI

**Expected:** DAG completes successfully
**Actual:** Task `train_xgboost` fails with OOM error

**Environment:**
- OS: macOS 14.0
- Python: 3.11.5
- Docker: 24.0.6

**Logs:**
```
[paste relevant logs]
```
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use case**: Why this enhancement would be useful
- **Proposed solution**: How you envision this working
- **Alternatives considered**: Other approaches you've considered
- **Implementation complexity**: Rough estimate of effort required

### Contributing Code

1. **Check existing issues** to see if your idea is already being discussed
2. **Create an issue** to discuss your proposed changes
3. **Fork the repository** and create a branch from `main`
4. **Make your changes** following our code standards
5. **Test thoroughly** - all tests must pass
6. **Submit a pull request** with a clear description

---

## Code Standards

### Python Style Guide

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: Maximum 100 characters
- **Import order**: Standard library → Third-party → Local
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Type hints**: Use type hints for function signatures

**Example:**
```python
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src_clean.utils.logger import setup_logger


def process_features(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """Process features for model training.

    Args:
        data: Input DataFrame containing features and target
        feature_columns: List of column names to use as features
        target_column: Optional target column name

    Returns:
        Dictionary containing processed features and labels

    Raises:
        ValueError: If feature_columns are not in data
    """
    # Implementation here
    pass
```

### File Organization

```
src_clean/
├── data_pipelines/      # Data processing pipelines
│   ├── bronze/          # Raw data ingestion
│   ├── silver/          # Feature engineering
│   └── gold/            # Training preparation
├── training/            # Model training scripts
├── monitoring/          # Drift detection & alerting
├── api/                 # FastAPI backend
├── ui/                  # Streamlit frontend
└── utils/               # Shared utilities
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

### Logging

Use the project's logging utility:

```python
from src_clean.utils.logger import setup_logger

logger = setup_logger(__name__)

logger.info("Processing started")
logger.warning("Missing values detected")
logger.error("Training failed", exc_info=True)
```

---

## Testing Requirements

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_features.py

# Run with coverage
make test-coverage
```

### Test Structure

Tests should be placed in the `tests/` directory:

```
tests/
├── unit/               # Unit tests for individual functions
├── integration/        # Integration tests for pipelines
└── e2e/                # End-to-end tests
```

### Writing Tests

Use `pytest` for all tests:

```python
import pytest
from src_clean.data_pipelines.silver import MarketTechnicalProcessor


def test_rsi_calculation():
    """Test RSI indicator calculation."""
    processor = MarketTechnicalProcessor()
    data = create_sample_ohlcv_data()

    result = processor.calculate_rsi(data, period=14)

    assert 'rsi_14' in result.columns
    assert result['rsi_14'].between(0, 100).all()
    assert not result['rsi_14'].isna().any()


@pytest.mark.integration
def test_full_silver_pipeline():
    """Test complete silver layer processing."""
    # Integration test implementation
    pass
```

### Test Coverage

- **Minimum coverage**: 70% for new code
- **Focus areas**: Data processing, feature engineering, model training
- **Mock external services**: Use `pytest-mock` for API calls, database connections

---

## Pull Request Process

### Before Submitting

1. **Update documentation** if you've changed functionality
2. **Add tests** for new features
3. **Run linters** and fix any issues:
   ```bash
   make lint  # Or: flake8 src_clean/ tests/
   ```
4. **Update CHANGELOG.md** with your changes
5. **Test locally** - ensure all tests pass and Docker services work

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No new warnings generated

## Related Issues
Closes #(issue number)
```

### Review Process

1. **Automated checks** must pass (tests, linting)
2. **Code review** by at least one maintainer
3. **Address feedback** promptly
4. **Squash commits** before merging (if requested)
5. **Delete branch** after merge

---

## Development Workflow

### Branch Naming

- **Features**: `feature/short-description`
- **Bug fixes**: `fix/short-description`
- **Hotfixes**: `hotfix/short-description`
- **Documentation**: `docs/short-description`

**Examples:**
- `feature/add-lstm-model`
- `fix/drift-detection-threshold`
- `docs/update-quickstart-guide`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(training): add LightGBM model support

Implement LightGBM training pipeline with 2-stage Optuna optimization.
Includes automatic model selection based on test RMSE.

Closes #42
```

```
fix(api): correct news path in inference engine

Update legacy path from data/news/gold to data_clean/gold/news/signals

Fixes #58
```

---

## Development Tips

### Useful Commands

```bash
# Format code
make format  # Or: black src_clean/ tests/

# Type checking
make typecheck  # Or: mypy src_clean/

# Run specific DAG
docker-compose exec airflow-scheduler airflow dags trigger sp500_ml_pipeline_v4_docker

# View logs
docker-compose logs -f <service-name>

# Access MLflow UI
open http://localhost:5001

# Access Airflow UI
open http://localhost:8080
```

### Debugging

1. **Docker logs**: `docker-compose logs -f <service>`
2. **Airflow logs**: Check Airflow UI → DAG → Task → Logs
3. **Application logs**: `logs/` directory
4. **Interactive debugging**: Use `pdb` or IDE debugger

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## Questions?

- **Documentation**: Check [docs/](docs/) directory
- **Issues**: Search [existing issues](https://github.com/kht321/fx-ml-pipeline/issues)
- **Discussions**: Start a [discussion](https://github.com/kht321/fx-ml-pipeline/discussions)

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

---

**Thank you for contributing!** 🚀
