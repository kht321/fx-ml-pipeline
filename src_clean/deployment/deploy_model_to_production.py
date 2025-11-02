from pathlib import Path
import shutil
import json
from datetime import datetime
import argparse
import sys
import pickle

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prod-path", required=True)
    args = parser.parse_args()

    print('=== Model Deployment (based on selection_info.json)===')

    # Load selection info
    production_dir = Path(args.prod_path)
    selection_file = production_dir / 'selection_info.json'

    if not selection_file.exists():
        print('ERROR: selection_info.json not found')
        sys.exit(1)

    with open(selection_file) as f:
        selection = json.load(f)

    model_type = selection['selected_model']
    latest_model_name = selection['selected_model_name'].replace('.pkl','')
    latest_model_path = Path(selection['model_path'])

    # Load pkl file
    if latest_model_path.exists():
        print(f"Found file. Loading: {latest_model_path}")
        with open(latest_model_path, 'rb') as f:
            latest_model = pickle.load(f)
    else:
        print(f"Error: File not found: {latest_model_path}")

    # Create production directory
    prod_dir = Path(args.prod_path)
    prod_dir.mkdir(exist_ok=True)

    # Copy model files to production
    prod_model = prod_dir / 'current_model.pkl'
    shutil.copy(latest_model_path, prod_model)
    print(f'✓ Deployed model: {latest_model_path} -> production/current_model.pkl')

    models_dir = Path(args.model_path + "/" + model_type)
    print(f'models_dir: {models_dir}')
    print(f'latest_model_name: {latest_model_name}')
    # Copy metrics if exists
    metrics_file = models_dir / f'{latest_model_name}_metrics.json'
    if metrics_file.exists():
        shutil.copy(metrics_file, prod_dir / 'current_metrics.json')
        print('✓ Deployed metrics')

    # # Copy features if exists
    features_file = models_dir / f'{latest_model_name}_features.json'
    if features_file.exists():
        shutil.copy(features_file, prod_dir / 'current_features.json')
        print('✓ Deployed feature definitions')

    # Create deployment metadata
    deployment_info = {
        'model_name': latest_model_path.name,
        'deployed_at': datetime.now().isoformat(),
        'model_size_mb': latest_model_path.stat().st_size / (1024 * 1024),
        'deployment_path': str(prod_model)
    }

    with open(prod_dir / 'deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)

    print('✓ Model deployed to production successfully!')
    print(f'Deployment info: {deployment_info}')


if __name__ == "__main__":
    main()