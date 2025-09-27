"""Train a logistic-regression baseline on Gold-layer features."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "training_path",
        nargs="?",
        type=Path,
        default=Path("data/gold/training/sgd_vs_majors_training.csv"),
        help="Gold-layer CSV containing features and target column",
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
        "--shuffle",
        action="store_true",
        help="Shuffle before splitting (default: chronological split)",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("data/gold/models/logreg_baseline.pkl"),
        help="Persist the fitted model (set to '-' to skip saving)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def prepare_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    df = df.copy()
    y = df[target_col].astype(np.int8)

    drop_cols = [
        target_col,
        "time",
        "published_at",
        "news_story_id",
        "news_headline",
        "news_source",
        "path",
    ]
    existing_drop = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=existing_drop)

    cat_cols = [col for col in X.columns if X[col].dtype == object]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.fillna(0.0)

    feature_names = list(X.columns)
    return X, y, feature_names


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.training_path.exists():
        raise SystemExit(f"Training data not found: {args.training_path}")

    df = pd.read_csv(args.training_path)

    X, y, feature_names = prepare_features(df, args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        shuffle=args.shuffle,
        stratify=y if args.shuffle else None,
        random_state=42,
    )

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)

    summary = {
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_names,
        "coefficients": dict(zip(feature_names, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0]),
        "classification_report": report,
    }

    print(json.dumps(summary, indent=2))

    if args.model_output and args.model_output != Path("-"):
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        dump({"model": model, "scaler": scaler, "features": feature_names}, args.model_output)


if __name__ == "__main__":
    main()
