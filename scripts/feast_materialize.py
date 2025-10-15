#!/usr/bin/env python3
"""Materialize features to Feast online store (Redis).

This script:
1. Applies feature definitions to Feast registry
2. Materializes historical features to the online store (Redis)
3. Enables real-time feature serving for model inference
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def log(message: str):
    """Simple logging."""
    print(f"[feast_materialize] {message}")


def run_command(cmd: list, cwd: Path = None):
    """Run a shell command and handle errors."""
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"❌ Error: {result.stderr}")
        return False

    if result.stdout:
        print(result.stdout)

    return True


def main():
    """Main execution."""
    feature_repo_path = project_root / "feature_repo"

    if not feature_repo_path.exists():
        log(f"❌ Feature repo not found at {feature_repo_path}")
        sys.exit(1)

    log("Step 1: Applying feature definitions to Feast registry")
    if not run_command(["feast", "apply"], cwd=feature_repo_path):
        log("❌ Failed to apply feature definitions")
        sys.exit(1)

    log("✓ Feature definitions applied")

    # Calculate time range for materialization
    # Materialize last 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    log(f"\nStep 2: Materializing features to online store (Redis)")
    log(f"  Time range: {start_date.isoformat()} to {end_date.isoformat()}")

    # Note: feast materialize-incremental is more efficient for ongoing updates
    # For first-time setup, we use materialize with a specific time range

    cmd = [
        "feast",
        "materialize",
        start_date.isoformat(),
        end_date.isoformat()
    ]

    if not run_command(cmd, cwd=feature_repo_path):
        log("❌ Failed to materialize features")
        log("Make sure:")
        log("  1. Redis is running (docker-compose up redis)")
        log("  2. Parquet files exist in data/market/gold/ and data/news/gold/")
        log("  3. Parquet files have 'event_timestamp' column")
        sys.exit(1)

    log("✓ Features materialized to online store")

    log("\n✅ Feast setup complete!")
    log("\nNext steps:")
    log("  - Start Redis: docker-compose up -d redis")
    log("  - Query features: feast.FeatureStore().get_online_features(...)")
    log("  - Incremental updates: feast materialize-incremental <end_date>")

    log("\nTo verify online features are available:")
    log("  python scripts/test_feast_online.py")


if __name__ == "__main__":
    main()
