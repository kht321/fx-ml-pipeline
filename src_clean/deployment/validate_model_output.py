from pathlib import Path
import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prod-path", required=True)
    args = parser.parse_args()
    
    print('=== Validating Model Selection Output ===')

    # Check production directory
    production_dir = Path(args.prod_path)
    if not production_dir.exists():
        print('ERROR: Production directory not found')
        sys.exit(1)

    # Check for selection info
    selection_file = production_dir / 'selection_info.json'
    if not selection_file.exists():
        print('ERROR: selection_info.json not found')
        sys.exit(1)

    with open(selection_file) as f:
        selection = json.load(f)

    print(f"✓ Selected Model: {selection['selected_model'].upper()}")
    print(f"✓ Test RMSE: {selection['test_rmse']:.4f}")
    print(f"✓ Test MAE: {selection.get('test_mae', 'N/A')}")
    print(f"✓ OOT RMSE: {selection.get('oot_rmse', 'N/A')}")

    # Check model file
    model_file = Path(selection['model_path'])
    if not model_file.exists():
        print(f'ERROR: Model file not found: {model_file}')
        sys.exit(1)

    size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f'✓ Model file: {model_file.name} ({size_mb:.2f} MB)')

    # Check for features file
    features_file = production_dir / f"best_model_{selection['selected_model']}_features.json"
    if features_file.exists():
        print(f'✓ Features file found')
    else:
        print('⚠ Features file not found')

    print('✓ Model validation complete!')

if __name__ == "__main__":
    main()