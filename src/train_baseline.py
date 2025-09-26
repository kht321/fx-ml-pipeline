"""Train a simple logistic regression baseline on engineered features."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "features_path",
        type=Path,
        default=Path("data/proc/features.csv"),
        nargs="?",
        help="CSV file containing engineered features with target column 'y'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features_path)

    X = df.drop(columns=["y"])
    y = df["y"].astype(np.int8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    summary = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "coefficients": dict(zip(X.columns, model.coef_[0].tolist())),
        "report": report,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
