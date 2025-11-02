import mlflow
import pickle
import json
from pathlib import Path
import sys
import argparse

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prod-path", required=True)
    parser.add_argument("--mlflow-uri", required=True)
    args = parser.parse_args()

    print('=== MLflow Model Registration (Best Selected Model) ===')

    # Load selection info
    production_dir = Path(args.prod_path)
    selection_file = production_dir / 'selection_info.json'

    if not selection_file.exists():
        print('ERROR: selection_info.json not found')
        sys.exit(1)

    with open(selection_file) as f:
        selection = json.load(f)

    selected_model = selection['selected_model']
    model_path = Path(selection['model_path'])

    print(f"Selected Model Type: {selected_model.upper()}")
    print(f"Test RMSE: {selection['test_rmse']:.4f}")

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f'✓ Model loaded: {type(model).__name__}')

    # Set MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_uri)
    print(f'✓ MLflow URI: {mlflow.get_tracking_uri()}')

    # Register model
    try:
        with mlflow.start_run(run_name=f'production_{selected_model}_best'):
            # Log based on model type
            if selected_model == 'xgboost':
                mlflow.xgboost.log_model(model, 'model', registered_model_name='sp500_best_model_production')
            elif selected_model == 'lightgbm':
                mlflow.lightgbm.log_model(model, 'model', registered_model_name='sp500_best_model_production')
            else:
                mlflow.sklearn.log_model(model, 'model', registered_model_name='sp500_best_model_production')

            # Log metrics from selection
            metrics_to_log = {
                'test_rmse': selection['test_rmse'],
                'test_mae': selection.get('test_mae'),
                'oot_rmse': selection.get('oot_rmse'),
                'oot_mae': selection.get('oot_mae')
            }

            for k, v in metrics_to_log.items():
                if v is not None:
                    mlflow.log_metric(k, v)

            mlflow.log_param('selected_model_type', selected_model)

            print(f'✓ Model registered successfully as sp500_best_model_production')

    except Exception as e:
        print(f'⚠ MLflow registration warning: {e}')
        print('✓ Continuing despite registration issue')


if __name__ == "__main__":
    main()