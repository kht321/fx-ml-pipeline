"""Train combined models using both Market and News Gold layer data.

This script merges the Gold-layer outputs from both medallion pipelines to create
sophisticated models that leverage both technical market features and news sentiment
signals for SGD FX prediction.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Expose CLI controls for combined model training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-features",
        type=Path,
        default=Path("data/market/gold/training/market_features.csv"),
        help="Gold-layer market features CSV",
    )
    parser.add_argument(
        "--news-signals",
        type=Path,
        default=Path("data/news/gold/news_signals/trading_signals.csv"),
        help="Gold-layer news signals CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/combined/models"),
        help="Directory for trained model outputs",
    )
    parser.add_argument(
        "--target",
        default="y",
        help="Target column name (default: y)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of rows to reserve for evaluation",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["logistic", "random_forest", "gradient_boosting", "all"],
        default=["all"],
        help="Models to train",
    )
    parser.add_argument(
        "--news-tolerance",
        default="6H",
        help="Max lookback window for joining news to market data (e.g. 6H, 2H)",
    )
    parser.add_argument(
        "--focus-currency",
        default="USD_SGD",
        help="Primary currency pair to focus on",
    )
    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Perform cross-validation evaluation",
    )
    return parser.parse_args(list(argv) if argv else None)


def load_gold_data(market_path: Path, news_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Gold layer data from both pipelines."""

    # Load market features
    if market_path.exists():
        market_df = pd.read_csv(market_path)
        market_df['time'] = pd.to_datetime(market_df['time'])
    else:
        market_df = pd.DataFrame()

    # Load news signals
    if news_path.exists():
        news_df = pd.read_csv(news_path)
        news_df['signal_time'] = pd.to_datetime(news_df['signal_time'])
    else:
        news_df = pd.DataFrame()

    return market_df, news_df


def merge_market_news_features(market_df: pd.DataFrame,
                              news_df: pd.DataFrame,
                              focus_currency: str = "USD_SGD",
                              news_tolerance: str = "6H") -> pd.DataFrame:
    """Merge market features with news signals using as-of join."""

    if market_df.empty:
        print("No market data available")
        return pd.DataFrame()

    # Filter market data for focus currency
    market_currency = market_df[market_df['instrument'] == focus_currency].copy()
    if market_currency.empty:
        print(f"No market data for {focus_currency}")
        return pd.DataFrame()

    # If no news data, return market-only features
    if news_df.empty:
        print("No news data - using market features only")
        return market_currency

    # Filter news for relevant currency
    currency_code = focus_currency.split('_')[1]  # Extract SGD from USD_SGD
    relevant_news = news_df[news_df['currency'] == currency_code].copy()

    if relevant_news.empty:
        print(f"No news data for {currency_code} - using market features only")
        return market_currency

    # Convert tolerance to timedelta
    tolerance = pd.Timedelta(news_tolerance)

    # Perform as-of merge
    print(f"Merging {len(market_currency)} market obs with {len(relevant_news)} news signals")

    # Sort both dataframes by time
    market_currency = market_currency.sort_values('time')
    relevant_news = relevant_news.sort_values('signal_time')

    # Rename columns to avoid conflicts
    news_columns = {
        'signal_time': 'news_time',
        'avg_sentiment': 'news_sentiment',
        'avg_directional': 'news_directional_signal',
        'signal_strength': 'news_signal_strength',
        'trading_signal': 'news_trading_signal',
        'article_count': 'news_article_count',
        'high_impact_count': 'news_high_impact_count',
        'quality_score': 'news_quality_score',
        'recent_sentiment': 'news_recent_sentiment',
        'time_decay': 'news_time_decay',
        'decayed_signal': 'news_decayed_signal'
    }

    relevant_news = relevant_news.rename(columns=news_columns)

    # Select news features for merging
    news_features = [
        'news_time', 'news_sentiment', 'news_directional_signal',
        'news_signal_strength', 'news_trading_signal', 'news_article_count',
        'news_high_impact_count', 'news_quality_score', 'news_recent_sentiment',
        'news_time_decay', 'news_decayed_signal', 'dominant_policy_tone',
        'signal_consensus', 'policy_consensus'
    ]

    # Keep only existing columns
    available_news_cols = [col for col in news_features if col in relevant_news.columns]
    news_for_merge = relevant_news[available_news_cols].copy()

    # Create a merged dataset using as-of join logic
    merged_rows = []

    for _, market_row in market_currency.iterrows():
        market_time = market_row['time']

        # Find the most recent news within tolerance
        news_cutoff = market_time - tolerance
        eligible_news = news_for_merge[
            (news_for_merge['news_time'] <= market_time) &
            (news_for_merge['news_time'] >= news_cutoff)
        ]

        if not eligible_news.empty:
            # Take the most recent news
            latest_news = eligible_news.iloc[-1]

            # Calculate age of news
            news_age_minutes = (market_time - latest_news['news_time']).total_seconds() / 60

            # Merge row
            merged_row = market_row.to_dict()
            for col in available_news_cols:
                if col != 'news_time':
                    merged_row[col] = latest_news[col]

            merged_row['news_age_minutes'] = news_age_minutes
            merged_row['news_available'] = 1

        else:
            # No news available - use neutral defaults
            merged_row = market_row.to_dict()
            for col in available_news_cols:
                if col != 'news_time':
                    if 'sentiment' in col or 'signal' in col:
                        merged_row[col] = 0.0
                    elif 'count' in col:
                        merged_row[col] = 0
                    elif col == 'dominant_policy_tone':
                        merged_row[col] = 'neutral'
                    else:
                        merged_row[col] = 0.0

            merged_row['news_age_minutes'] = np.nan
            merged_row['news_available'] = 0

        merged_rows.append(merged_row)

    combined_df = pd.DataFrame(merged_rows)

    print(f"Combined dataset: {len(combined_df)} observations")
    news_coverage = combined_df['news_available'].mean()
    print(f"News coverage: {news_coverage:.1%} of market observations have news signals")

    return combined_df


def prepare_features(df: pd.DataFrame, target_col: str = 'y') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare features for model training."""

    if df.empty or target_col not in df.columns:
        return pd.DataFrame(), pd.Series(), []

    # Remove non-feature columns
    non_feature_cols = {
        'time', 'instrument', 'news_time', 'news_available',
        'latest_headline', 'latest_source', 'currency', 'lookback_hours',
        'signal_category', 'minutes_since_latest', 'dominant_time_horizon'
    }

    feature_cols = [col for col in df.columns
                   if col not in non_feature_cols and col != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if col in X.columns:
            # Simple label encoding for policy tone
            if 'policy' in col.lower():
                policy_map = {'hawkish': 1, 'neutral': 0, 'dovish': -1}
                X[col] = X[col].map(policy_map).fillna(0)
            else:
                # One-hot encoding for other categoricals
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)

    # Handle missing values
    X = X.fillna(0)

    # Remove constant columns
    constant_cols = X.columns[X.nunique() <= 1]
    if len(constant_cols) > 0:
        print(f"Removing {len(constant_cols)} constant columns")
        X = X.drop(constant_cols, axis=1)

    # Remove highly correlated features
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    high_corr_pairs = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > 0.95)
    ]

    if high_corr_pairs:
        print(f"Removing {len(high_corr_pairs)} highly correlated features")
        X = X.drop(high_corr_pairs, axis=1)

    print(f"Final feature set: {len(X.columns)} features")

    return X, y, list(X.columns)


def train_models(X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series,
                model_types: List[str]) -> Dict:
    """Train multiple model types and return results."""

    if "all" in model_types:
        model_types = ["logistic", "random_forest", "gradient_boosting"]

    # Initialize models
    models = {}
    if "logistic" in model_types:
        models["logistic"] = LogisticRegression(random_state=42, max_iter=1000)
    if "random_forest" in model_types:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    if "gradient_boosting" in model_types:
        models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=100, random_state=42
        )

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Use scaled features for logistic regression
        if name == "logistic":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate model
        auc_score = roc_auc_score(y_test, y_pred_proba)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            'model': model,
            'scaler': scaler if name == "logistic" else None,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        print(f"{name} AUC: {auc_score:.3f}")
        print(f"{name} Accuracy: {class_report['accuracy']:.3f}")

        # Feature importance for tree models
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            results[name]['feature_importance'] = feature_importance.head(20).to_dict('records')

    return results


def cross_validate_models(X: pd.DataFrame, y: pd.Series, model_types: List[str]) -> Dict:
    """Perform cross-validation on models."""

    if "all" in model_types:
        model_types = ["logistic", "random_forest", "gradient_boosting"]

    models = {}
    if "logistic" in model_types:
        models["logistic"] = LogisticRegression(random_state=42, max_iter=1000)
    if "random_forest" in model_types:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    if "gradient_boosting" in model_types:
        models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=100, random_state=42
        )

    cv_results = {}

    for name, model in models.items():
        print(f"\nCross-validating {name}...")

        if name == "logistic":
            # Scale features for logistic regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
        else:
            scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

        cv_results[name] = {
            'mean_auc': scores.mean(),
            'std_auc': scores.std(),
            'scores': scores.tolist()
        }

        print(f"{name} CV AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    return cv_results


def save_models_and_results(results: Dict,
                           feature_names: List[str],
                           output_dir: Path,
                           cv_results: Dict = None):
    """Save trained models and evaluation results."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, result in results.items():
        # Save model
        model_path = output_dir / f"{model_name}_combined_model.pkl"

        model_bundle = {
            'model': result['model'],
            'scaler': result['scaler'],
            'feature_names': feature_names,
            'model_type': model_name
        }

        dump(model_bundle, model_path)

        # Save metrics
        metrics = {
            'model_name': model_name,
            'auc_score': result['auc_score'],
            'classification_report': result['classification_report'],
            'confusion_matrix': result['confusion_matrix']
        }

        if 'feature_importance' in result:
            metrics['feature_importance'] = result['feature_importance']

        if cv_results and model_name in cv_results:
            metrics['cross_validation'] = cv_results[model_name]

        metrics_path = output_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    print(f"\nModels and results saved to {output_dir}")


def log(message: str) -> None:
    """Emit structured progress messages."""
    print(f"[train_combined_model] {message}")


def main(argv: Iterable[str] | None = None) -> None:
    """Main training function for combined models."""
    args = parse_args(argv)

    log("Loading Gold layer data from both pipelines")

    # Load data
    market_df, news_df = load_gold_data(args.market_features, args.news_signals)

    if market_df.empty:
        log("No market data available - cannot proceed")
        return

    log(f"Loaded: {len(market_df)} market observations, {len(news_df)} news signals")

    # Merge market and news features
    log("Merging market and news features")
    combined_df = merge_market_news_features(
        market_df, news_df, args.focus_currency, args.news_tolerance
    )

    if combined_df.empty:
        log("No combined data available - cannot proceed")
        return

    # Prepare features
    log("Preparing features for modeling")
    X, y, feature_names = prepare_features(combined_df, args.target)

    if X.empty or len(y) == 0:
        log("No valid features or targets - cannot proceed")
        return

    log(f"Training dataset: {len(X)} observations, {len(feature_names)} features")
    log(f"Target distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    log(f"Split: {len(X_train)} train, {len(X_test)} test")

    # Cross-validation if requested
    cv_results = None
    if args.cross_validation:
        log("Performing cross-validation")
        cv_results = cross_validate_models(X, y, args.models)

    # Train models
    log("Training models")
    results = train_models(X_train, X_test, y_train, y_test, args.models)

    # Save results
    save_models_and_results(results, feature_names, args.output_dir, cv_results)

    # Summary
    log("Training complete. Model performance:")
    for name, result in results.items():
        log(f"  {name}: AUC = {result['auc_score']:.3f}, "
            f"Accuracy = {result['classification_report']['accuracy']:.3f}")


if __name__ == "__main__":
    main()