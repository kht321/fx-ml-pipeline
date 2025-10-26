"""
Optimized XGBoost Configuration for S&P 500 Prediction

Implements aggressive regularization and optimized parameters to reduce overfitting
and improve generalization for 52-53% AUC target.

Repository Location: fx-ml-pipeline/src_clean/training/xgboost_optimized_config.py
"""

# Optimized parameters based on analysis
OPTIMIZED_PARAMS = {
    "classification": {
        # Core parameters
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",  # Fast histogram-based method

        # Reduced complexity to prevent overfitting
        "max_depth": 3,  # Reduced from 6 (shallower trees)
        "n_estimators": 100,  # Reduced from 200
        "learning_rate": 0.01,  # Reduced from 0.1 (slower learning)

        # Aggressive regularization
        "reg_alpha": 1.0,  # L1 regularization (was 0)
        "reg_lambda": 2.0,  # L2 regularization (was 1)
        "gamma": 1.0,  # Minimum loss reduction (was 0)

        # Sampling to reduce overfitting
        "subsample": 0.5,  # More aggressive subsampling (was 0.8)
        "colsample_bytree": 0.5,  # Fewer features per tree (was 0.8)
        "colsample_bylevel": 0.5,  # Fewer features per level (new)

        # Other parameters
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,

        # Early stopping
        "early_stopping_rounds": 50,
        "eval_metric": ["auc", "logloss"]
    },

    "regression": {
        # Core parameters
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",

        # Reduced complexity
        "max_depth": 3,
        "n_estimators": 100,
        "learning_rate": 0.01,

        # Aggressive regularization
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "gamma": 1.0,

        # Sampling
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "colsample_bylevel": 0.5,

        # Other
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 1,

        # Early stopping
        "early_stopping_rounds": 50,
        "eval_metric": ["rmse", "mae"]
    }
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    "models": [
        {
            "name": "xgboost_conservative",
            "params": {
                **OPTIMIZED_PARAMS["classification"],
                "max_depth": 2,
                "n_estimators": 50,
                "learning_rate": 0.005
            },
            "weight": 0.3
        },
        {
            "name": "xgboost_moderate",
            "params": OPTIMIZED_PARAMS["classification"],
            "weight": 0.4
        },
        {
            "name": "xgboost_aggressive",
            "params": {
                **OPTIMIZED_PARAMS["classification"],
                "max_depth": 4,
                "n_estimators": 150,
                "learning_rate": 0.02
            },
            "weight": 0.3
        }
    ]
}

# Feature selection configuration
FEATURE_SELECTION = {
    "method": "importance_threshold",
    "threshold": 0.001,  # Remove features with < 0.1% importance
    "max_features": 50,  # Keep only top 50 features
    "correlation_threshold": 0.95  # Remove highly correlated features
}

# Cross-validation configuration
CV_CONFIG = {
    "n_splits": 10,  # More folds for better validation
    "gap_size": 100,  # Gap between train and test to prevent leakage
    "test_size": 10000,  # Fixed test size for each fold
    "purge_ratio": 0.01  # Purge 1% of data around split points
}

# Threshold tuning
THRESHOLD_CONFIG = {
    "optimize_threshold": True,
    "metric": "f1_score",  # Optimize for F1 instead of accuracy
    "search_range": (0.45, 0.55),  # Search around 0.5
    "n_points": 100  # Number of thresholds to test
}

def get_optimized_params(task="classification", features_count=70):
    """
    Get optimized parameters based on task and feature count.

    Parameters
    ----------
    task : str
        'classification' or 'regression'
    features_count : int
        Number of features in dataset

    Returns
    -------
    dict
        Optimized XGBoost parameters
    """
    params = OPTIMIZED_PARAMS[task].copy()

    # Adjust based on feature count
    if features_count > 100:
        # More features = need more regularization
        params["reg_alpha"] = 2.0
        params["reg_lambda"] = 3.0
        params["colsample_bytree"] = 0.3
    elif features_count < 30:
        # Fewer features = can be less aggressive
        params["reg_alpha"] = 0.5
        params["reg_lambda"] = 1.0
        params["colsample_bytree"] = 0.7

    return params


def get_lightgbm_optimized_params(task="classification"):
    """Get optimized LightGBM parameters."""
    return {
        "classification": {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",

            # Reduced complexity
            "num_leaves": 15,  # Reduced from 31
            "max_depth": 3,
            "learning_rate": 0.01,
            "n_estimators": 100,

            # Regularization
            "lambda_l1": 1.0,
            "lambda_l2": 2.0,
            "min_gain_to_split": 1.0,
            "min_child_weight": 10,

            # Sampling
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,

            # Other
            "random_state": 42,
            "verbosity": -1,
            "num_threads": -1
        },

        "regression": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",

            # Reduced complexity
            "num_leaves": 15,
            "max_depth": 3,
            "learning_rate": 0.01,
            "n_estimators": 100,

            # Regularization
            "lambda_l1": 1.0,
            "lambda_l2": 2.0,
            "min_gain_to_split": 1.0,
            "min_child_weight": 10,

            # Sampling
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,

            # Other
            "random_state": 42,
            "verbosity": -1,
            "num_threads": -1
        }
    }[task]


if __name__ == "__main__":
    print("Optimized XGBoost Configuration")
    print("="*60)
    print("\nClassification parameters:")
    for key, value in OPTIMIZED_PARAMS["classification"].items():
        print(f"  {key}: {value}")

    print("\nKey changes from default:")
    print("  - max_depth: 6 → 3 (shallower trees)")
    print("  - learning_rate: 0.1 → 0.01 (slower learning)")
    print("  - subsample: 0.8 → 0.5 (more aggressive)")
    print("  - reg_alpha: 0 → 1.0 (L1 regularization)")
    print("  - reg_lambda: 1 → 2.0 (L2 regularization)")
    print("\nExpected improvement: 1-2% AUC from regularization alone")