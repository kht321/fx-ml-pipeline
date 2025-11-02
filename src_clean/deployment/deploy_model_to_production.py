from pathlib import Path
import shutil
import json
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prod-path", required=True)
    args = parser.parse_args()

    print('=== Model Deployment ===')

    models_dir = Path(args.model_path)
    pkl_files = sorted(models_dir.glob('xgboost_regression_*.pkl'), key=lambda x: x.stat().st_mtime, reverse=True)

    if not pkl_files:
        print('ERROR: No models to deploy')
        exit(1)

    latest_model = pkl_files[0]
    model_base = latest_model.stem

    # Create production directory
    prod_dir = Path(args.prod_path)
    prod_dir.mkdir(exist_ok=True)

    # Copy model files to production
    prod_model = prod_dir / 'current_model.pkl'
    shutil.copy(latest_model, prod_model)
    print(f'✓ Deployed model: {latest_model.name} -> production/current_model.pkl')

    # Copy metrics if exists
    metrics_file = models_dir / f'{model_base}_metrics.json'
    if metrics_file.exists():
        shutil.copy(metrics_file, prod_dir / 'current_metrics.json')
        print('✓ Deployed metrics')

    # Copy features if exists
    features_file = models_dir / f'{model_base}_features.json'
    if features_file.exists():
        shutil.copy(features_file, prod_dir / 'current_features.json')
        print('✓ Deployed feature definitions')

    # Create deployment metadata
    deployment_info = {
        'model_name': latest_model.name,
        'deployed_at': datetime.utcnow().isoformat(),
        'model_size_mb': latest_model.stat().st_size / (1024 * 1024),
        'deployment_path': str(prod_model)
    }

    with open(prod_dir / 'deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)

    print('✓ Model deployed to production successfully!')
    print(f'Deployment info: {deployment_info}')


if __name__ == "__main__":
    main()