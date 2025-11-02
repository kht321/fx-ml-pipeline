import json
import shutil
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    print('=== Model Selection: Comparing XGBoost, LightGBM, ARIMAX, AR ===')

    models_base = Path(args.model_path)
    model_types = ['xgboost', 'lightgbm', 'arima']

    # Find metrics for each model
    best_model = None
    best_rmse = float('inf')
    best_metrics = None
    best_model_path = None

    for model_type in model_types:
        model_dir = models_base / model_type
        if not model_dir.exists():
            print(f'⚠ {model_type} directory not found, skipping')
            continue

        # Find latest metrics file
        metrics_files = sorted(model_dir.glob('*_metrics.json'),
                            key=lambda x: x.stat().st_mtime,
                            reverse=True)

        if not metrics_files:
            print(f'⚠ No metrics found for {model_type}')
            continue

        metrics_path = metrics_files[0]
        with open(metrics_path) as f:
            metrics = json.load(f)

        # Get test RMSE
        test_rmse = metrics.get('test_rmse', float('inf'))
        test_mae = metrics.get('test_mae', float('inf'))

        print(f'{model_type.upper()}: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}')

        # Track best model
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model = model_type
            best_metrics = metrics

            # Find model file
            if model_type == 'arima':
                model_files = sorted(model_dir.glob('arima_*.pkl'),
                                key=lambda x: x.stat().st_mtime,
                                reverse=True)
            else:
                model_files = sorted(model_dir.glob(f'{model_type}_regression_*.pkl'),
                                key=lambda x: x.stat().st_mtime,
                                reverse=True)

            if model_files:
                best_model_path = model_files[0]

    if best_model is None:
        print('ERROR: No models found!')
        exit(1)

    print(f'\\n✅ BEST MODEL: {best_model.upper()} (RMSE={best_rmse:.4f})')

    # Copy best model to production directory
    production_dir = models_base / 'production'
    production_dir.mkdir(exist_ok=True)

    if best_model_path:
        # Copy model file
        prod_model_path = production_dir / f'best_model_{best_model}.pkl'
        shutil.copy2(best_model_path, prod_model_path)
        print(f'✓ Copied {best_model_path.name} -> {prod_model_path.name}')

        # Copy features file
        features_src = best_model_path.parent / f'{best_model_path.stem}_features.json'
        if features_src.exists():
            features_dst = production_dir / f'best_model_{best_model}_features.json'
            shutil.copy2(features_src, features_dst)
            print(f'✓ Copied features file')

        # Save selection metadata
        selection_info = {
            'selected_model': best_model,
            'selected_model_name': best_model_path.name,
            'test_rmse': best_rmse,
            'test_mae': best_metrics.get('test_mae', None),
            'oot_rmse': best_metrics.get('oot_rmse', None),
            'oot_mae': best_metrics.get('oot_mae', None),
            'model_path': str(prod_model_path),
        }

        with open(production_dir / 'selection_info.json', 'w') as f:
            json.dump(selection_info, f, indent=2)

        print('✓ Model selection complete!')
    else:
        print('ERROR: Could not find model file for best model')
        exit(1)


if __name__ == "__main__":
    main()