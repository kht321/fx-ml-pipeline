"""
Generate presentation-ready visualizations for model training results.

This script creates comprehensive charts for model evaluation including:
- Model comparison (RMSE, MAE across train/val/test/OOT)
- Predictions vs Actuals scatter plots
- Residual analysis
- Feature importance
- Performance metrics dashboard
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model directories
MODELS_DIR = Path("models")


def load_latest_metrics():
    """Load metrics from all latest model runs."""
    metrics_data = {}

    # XGBoost metrics
    xgb_metrics_files = sorted(
        MODELS_DIR.glob("xgboost/**/xgboost_regression_*_metrics.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if xgb_metrics_files:
        with open(xgb_metrics_files[0]) as f:
            metrics_data['XGBoost'] = json.load(f)

    # LightGBM metrics
    lgbm_metrics_files = sorted(
        MODELS_DIR.glob("lightgbm/**/lightgbm_regression_*_metrics.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if lgbm_metrics_files:
        with open(lgbm_metrics_files[0]) as f:
            metrics_data['LightGBM'] = json.load(f)

    # AR metrics
    ar_metrics_files = sorted(
        MODELS_DIR.glob("ar/**/ar_regression_*_metrics.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if ar_metrics_files:
        with open(ar_metrics_files[0]) as f:
            ar_metrics = json.load(f)
            # Normalize AR metrics keys
            metrics_data['ARIMAX'] = {
                'train_rmse': ar_metrics.get('train_rmse_t30', 0),
                'train_mae': ar_metrics.get('train_mae_t30', 0),
                'val_rmse': ar_metrics.get('val_rmse_t30', 0),
                'val_mae': ar_metrics.get('val_mae_t30', 0),
                'test_rmse': ar_metrics.get('test_rmse_t30', 0),
                'test_mae': ar_metrics.get('test_mae_t30', 0),
                'oot_rmse': ar_metrics.get('oot_rmse_t30', 0),
                'oot_mae': ar_metrics.get('oot_mae_t30', 0),
                'n_features': ar_metrics.get('n_features', 0),
            }

    return metrics_data


def plot_model_comparison(metrics_data):
    """Create comprehensive model comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison: 30-Minute Price Prediction',
                 fontsize=20, fontweight='bold', y=0.995)

    # Prepare data
    models = list(metrics_data.keys())

    # RMSE comparison
    train_rmse = [metrics_data[m].get('train_rmse', 0) for m in models]
    val_rmse = [metrics_data[m].get('val_rmse', 0) for m in models]
    test_rmse = [metrics_data[m].get('test_rmse', 0) for m in models]
    oot_rmse = [metrics_data[m].get('oot_rmse', 0) for m in models]

    # MAE comparison
    train_mae = [metrics_data[m].get('train_mae', 0) for m in models]
    val_mae = [metrics_data[m].get('val_mae', 0) for m in models]
    test_mae = [metrics_data[m].get('test_mae', 0) for m in models]
    oot_mae = [metrics_data[m].get('oot_mae', 0) for m in models]

    x = np.arange(len(models))
    width = 0.2

    # RMSE plot
    ax = axes[0, 0]
    ax.bar(x - 1.5*width, train_rmse, width, label='Train', color='#2ecc71')
    ax.bar(x - 0.5*width, val_rmse, width, label='Validation', color='#3498db')
    ax.bar(x + 0.5*width, test_rmse, width, label='Test', color='#e74c3c')
    ax.bar(x + 1.5*width, oot_rmse, width, label='OOT', color='#f39c12')
    ax.set_ylabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Root Mean Squared Error Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # MAE plot
    ax = axes[0, 1]
    ax.bar(x - 1.5*width, train_mae, width, label='Train', color='#2ecc71')
    ax.bar(x - 0.5*width, val_mae, width, label='Validation', color='#3498db')
    ax.bar(x + 0.5*width, test_mae, width, label='Test', color='#e74c3c')
    ax.bar(x + 1.5*width, oot_mae, width, label='OOT', color='#f39c12')
    ax.set_ylabel('MAE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Test performance comparison
    ax = axes[1, 0]
    test_metrics = pd.DataFrame({
        'Model': models,
        'RMSE': test_rmse,
        'MAE': test_mae
    }).set_index('Model')
    test_metrics.plot(kind='bar', ax=ax, width=0.7, color=['#e74c3c', '#3498db'])
    ax.set_ylabel('Error Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # OOT performance comparison
    ax = axes[1, 1]
    oot_metrics = pd.DataFrame({
        'Model': models,
        'RMSE': oot_rmse,
        'MAE': oot_mae
    }).set_index('Model')
    oot_metrics.plot(kind='bar', ax=ax, width=0.7, color=['#f39c12', '#9b59b6'])
    ax.set_ylabel('Error Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Out-of-Time (OOT) Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'model_comparison.png'}")
    plt.close()


def plot_metrics_dashboard(metrics_data):
    """Create a comprehensive metrics dashboard."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Model Performance Dashboard - S&P 500 30-Minute Price Prediction',
                 fontsize=20, fontweight='bold', y=0.98)

    models = list(metrics_data.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # 1. Overall RMSE Comparison (spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    metrics_df = pd.DataFrame({
        model: [
            metrics_data[model].get('train_rmse', 0),
            metrics_data[model].get('val_rmse', 0),
            metrics_data[model].get('test_rmse', 0),
            metrics_data[model].get('oot_rmse', 0)
        ]
        for model in models
    }, index=['Train', 'Validation', 'Test', 'OOT'])

    metrics_df.T.plot(kind='bar', ax=ax1, color=colors[:len(models)], width=0.8)
    ax1.set_title('RMSE Across All Datasets', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=11)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.legend(title='Dataset', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Model Summary Table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    table_data = []
    for model in models:
        table_data.append([
            model,
            f"{metrics_data[model].get('test_rmse', 0):.4f}",
            f"{metrics_data[model].get('oot_rmse', 0):.4f}",
            f"{metrics_data[model].get('n_features', 'N/A')}"
        ])

    table = ax2.table(cellText=table_data,
                     colLabels=['Model', 'Test RMSE', 'OOT RMSE', 'Features'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    for i in range(len(table_data) + 1):
        if i == 0:
            for j in range(4):
                table[(i, j)].set_facecolor('#34495e')
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            for j in range(4):
                table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

    ax2.set_title('Model Summary', fontsize=12, fontweight='bold', pad=20)

    # 3. MAE Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    mae_data = {model: metrics_data[model].get('test_mae', 0) for model in models}
    ax3.barh(list(mae_data.keys()), list(mae_data.values()), color=colors[:len(models)])
    ax3.set_xlabel('Test MAE', fontsize=11)
    ax3.set_title('Test MAE Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # 4. OOT Performance
    ax4 = fig.add_subplot(gs[1, 1])
    oot_data = {model: metrics_data[model].get('oot_mae', 0) for model in models}
    ax4.barh(list(oot_data.keys()), list(oot_data.values()), color=colors[:len(models)])
    ax4.set_xlabel('OOT MAE', fontsize=11)
    ax4.set_title('OOT MAE Comparison', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    # 5. Train vs Val RMSE
    ax5 = fig.add_subplot(gs[1, 2])
    train_rmse = [metrics_data[m].get('train_rmse', 0) for m in models]
    val_rmse = [metrics_data[m].get('val_rmse', 0) for m in models]

    x = np.arange(len(models))
    width = 0.35
    ax5.bar(x - width/2, train_rmse, width, label='Train', color='#2ecc71')
    ax5.bar(x + width/2, val_rmse, width, label='Validation', color='#3498db')
    ax5.set_ylabel('RMSE', fontsize=11)
    ax5.set_title('Train vs Validation', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(models, fontsize=9)
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)

    # 6-8. Individual model performance breakdown
    for idx, model in enumerate(models):
        ax = fig.add_subplot(gs[2, idx])

        datasets = ['Train', 'Val', 'Test', 'OOT']
        rmse_vals = [
            metrics_data[model].get('train_rmse', 0),
            metrics_data[model].get('val_rmse', 0),
            metrics_data[model].get('test_rmse', 0),
            metrics_data[model].get('oot_rmse', 0)
        ]
        mae_vals = [
            metrics_data[model].get('train_mae', 0),
            metrics_data[model].get('val_mae', 0),
            metrics_data[model].get('test_mae', 0),
            metrics_data[model].get('oot_mae', 0)
        ]

        x = np.arange(len(datasets))
        width = 0.35
        ax.bar(x - width/2, rmse_vals, width, label='RMSE', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, mae_vals, width, label='MAE', color='#3498db', alpha=0.8)
        ax.set_ylabel('Error', fontsize=10)
        ax.set_title(f'{model} Performance', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.savefig(OUTPUT_DIR / 'metrics_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'metrics_dashboard.png'}")
    plt.close()


def plot_feature_importance():
    """Plot feature importance from XGBoost model."""
    # Find latest XGBoost metrics with feature importances
    xgb_metrics_files = sorted(
        MODELS_DIR.glob("xgboost/**/xgboost_regression_*_metrics.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not xgb_metrics_files:
        print("⚠ No XGBoost metrics found for feature importance")
        return

    with open(xgb_metrics_files[0]) as f:
        metrics = json.load(f)

    features = metrics.get('selected_features', [])

    if not features:
        print("⚠ No feature information in metrics")
        return

    # Create feature categories
    feature_categories = {
        'Price & Volume': ['open', 'high', 'low', 'close', 'volume'],
        'Returns': [f for f in features if 'return_' in f],
        'Technical Indicators': [f for f in features if any(x in f for x in ['rsi', 'macd', 'bb_', 'sma_', 'ema_', 'atr', 'adx'])],
        'Volatility': [f for f in features if 'vol' in f.lower() or f in ['gk_vol', 'parkinson_vol', 'rs_vol', 'yz_vol']],
        'Volume Metrics': [f for f in features if 'volume_' in f and f not in ['volume', 'volume_ma20', 'volume_ma50']],
        'Microstructure': [f for f in features if any(x in f for x in ['price_impact', 'order_flow', 'illiquidity', 'vwap', 'spread'])],
        'News Signals': [f for f in features if 'news_' in f],
        'Other': [f for f in features if not any(f in cat for cat in [
            ['open', 'high', 'low', 'close', 'volume'],
            [x for x in features if 'return_' in x],
            [x for x in features if any(y in x for y in ['rsi', 'macd', 'bb_', 'sma_', 'ema_', 'atr', 'adx'])],
            [x for x in features if 'vol' in x.lower()],
            [x for x in features if 'volume_' in x],
            [x for x in features if any(y in x for y in ['price_impact', 'order_flow', 'illiquidity', 'vwap', 'spread'])],
            [x for x in features if 'news_' in x]
        ])]
    }

    # Count features by category
    category_counts = {cat: len(feats) for cat, feats in feature_categories.items() if feats}

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Feature Analysis - XGBoost Model ({len(features)} Total Features)',
                 fontsize=18, fontweight='bold')

    # Category distribution
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(categories)))

    ax1.barh(categories, counts, color=colors_palette)
    ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Distribution by Category', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add counts on bars
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax1.text(count + 0.5, i, str(count), va='center', fontsize=10, fontweight='bold')

    # Pie chart
    ax2.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors_palette,
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Feature Category Proportions', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'feature_analysis.png'}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Model Evaluation Visualizations")
    print("=" * 60)

    # Load metrics
    print("\n1. Loading model metrics...")
    metrics_data = load_latest_metrics()

    if not metrics_data:
        print("❌ No model metrics found!")
        return

    print(f"   ✓ Found metrics for {len(metrics_data)} models: {', '.join(metrics_data.keys())}")

    # Generate visualizations
    print("\n2. Generating model comparison chart...")
    plot_model_comparison(metrics_data)

    print("\n3. Generating metrics dashboard...")
    plot_metrics_dashboard(metrics_data)

    print("\n4. Generating feature analysis...")
    plot_feature_importance()

    print("\n" + "=" * 60)
    print(f"✓ All visualizations saved to: {OUTPUT_DIR}/")
    print("=" * 60)

    # Print summary
    print("\nModel Performance Summary:")
    print("-" * 60)
    for model, metrics in metrics_data.items():
        print(f"\n{model}:")
        print(f"  Test RMSE:  {metrics.get('test_rmse', 0):.4f}")
        print(f"  Test MAE:   {metrics.get('test_mae', 0):.4f}")
        print(f"  OOT RMSE:   {metrics.get('oot_rmse', 0):.4f}")
        print(f"  OOT MAE:    {metrics.get('oot_mae', 0):.4f}")
        print(f"  Features:   {metrics.get('n_features', 'N/A')}")
    print("-" * 60)


if __name__ == "__main__":
    main()
