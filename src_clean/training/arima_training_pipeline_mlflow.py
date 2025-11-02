"""
ARIMA Training Pipeline with MLflow Integration - 30-Minute Price Prediction

This pipeline uses Auto-ARIMA with exogenous variables (ARIMAX) that includes:
- Market features and their lags
- News sentiment signals and their lags
The approach enforces stationarity on the target series and prevents data
leakage by respecting time-order splits throughout training and evaluation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ARIMAMLflowTrainingPipeline:
    """Training pipeline for Auto-ARIMA with MLflow tracking."""

    def __init__(
        self,
        market_features_path: Path,
        labels_path: Optional[Path] = None,
        news_signals_path: Optional[Path] = None,
        prediction_horizon_minutes: int = 30,
        output_dir: Path = Path("data_clean/models"),
        task: str = "regression",
        experiment_name: str = "sp500_prediction",
        max_lag: int = 3,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        stationarity_alpha: float = 0.05
    ):
        """
        Initialize the ARIMA training pipeline.

        Parameters
        ----------
        market_features_path : Path
            Path to Gold layer market features CSV.
        labels_path : Path, optional
            Path to Gold layer labels CSV (if not provided, will be inferred).
        news_signals_path : Path, optional
            Path to Gold layer news signals CSV.
        prediction_horizon_minutes : int
            Number of minutes ahead to predict (default: 30).
        output_dir : Path
            Directory to save trained models and outputs.
        task : str
            Only "regression" is supported for ARIMA.
        experiment_name : str
            MLflow experiment name.
        max_lag : int
            Maximum lag order to create for exogenous features.
        max_p : int
            Maximum AR order for auto ARIMA search.
        max_q : int
            Maximum MA order for auto ARIMA search.
        max_d : int
            Maximum differencing order evaluated for stationarity.
        stationarity_alpha : float
            Significance level for the ADF stationarity test.
        """
        if task != "regression":
            raise ValueError("ARIMA training pipeline currently supports regression tasks only.")

        self.market_features_path = market_features_path
        self.labels_path = labels_path
        self.news_signals_path = news_signals_path
        self.prediction_horizon = prediction_horizon_minutes
        self.output_dir = output_dir
        self.task = task
        self.experiment_name = experiment_name
        self.max_lag = max_lag
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.stationarity_alpha = stationarity_alpha

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load market features, labels, and news signals from Gold layer."""
        logger.info("Loading market features from Gold layer...")

        # Load market features (support both CSV and Parquet)
        if str(self.market_features_path).endswith('.parquet'):
            market_df = pd.read_parquet(self.market_features_path)
        else:
            market_df = pd.read_csv(self.market_features_path)
        market_df["time"] = pd.to_datetime(market_df["time"])
        market_df = market_df.sort_values("time").reset_index(drop=True)
        logger.info("Loaded market features: %d rows, %d columns", len(market_df), len(market_df.columns))

        if self.labels_path is None:
            features_dir = self.market_features_path.parent
            base_dir = features_dir.parent.parent.parent
            instrument = self.market_features_path.stem.replace("_features", "")
            inferred_labels = base_dir / f"gold/market/labels/{instrument}_labels_{self.prediction_horizon}min.csv"
            self.labels_path = inferred_labels
            logger.info("Inferred labels path: %s", self.labels_path)

        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}\n"
                "Please generate labels first or provide the --labels argument."
            )

        labels_df = pd.read_csv(self.labels_path)
        labels_df["time"] = pd.to_datetime(labels_df["time"])
        labels_df = labels_df.sort_values("time").reset_index(drop=True)
        logger.info("Loaded labels: %d rows, %d columns", len(labels_df), len(labels_df.columns))

        if "prediction_horizon_minutes" in labels_df.columns:
            label_horizon = labels_df["prediction_horizon_minutes"].iloc[0]
            if pd.notna(label_horizon) and int(label_horizon) != self.prediction_horizon:
                logger.warning(
                    "Label horizon (%s min) differs from requested horizon (%s min). Using label horizon.",
                    label_horizon,
                    self.prediction_horizon
                )
                self.prediction_horizon = int(label_horizon)

        label_columns = ["time", "target_pct_change"]
        if "fold" in labels_df.columns:
            label_columns.append("fold")

        merged_df = pd.merge(
            market_df,
            labels_df[label_columns],
            on="time",
            how="inner"
        )

        if merged_df.empty:
            raise ValueError("Merging market features with labels resulted in an empty dataframe.")

        merged_df = merged_df.sort_values("time").reset_index(drop=True)
        merged_df["target"] = merged_df["target_pct_change"]

        logger.info("Merged dataset: %d rows after join", len(merged_df))

        # Load news signals if available (support both CSV and Parquet)
        news_df = None
        if self.news_signals_path and self.news_signals_path.exists():
            if str(self.news_signals_path).endswith('.parquet'):
                news_df = pd.read_parquet(self.news_signals_path)
            else:
                news_df = pd.read_csv(self.news_signals_path)
            news_df['signal_time'] = pd.to_datetime(news_df['signal_time'])
            logger.info(f"Loaded news signals: {len(news_df)} rows, {len(news_df.columns)} columns")
        else:
            logger.warning("No news signals provided - using market features only")

        return merged_df, news_df

    def validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that labels are present and log statistics."""
        logger.info("Validating labels for %s-minute prediction...", self.prediction_horizon)

        if "target" not in df.columns:
            raise ValueError("Target column not found in dataframe. Labels should be loaded from Gold layer.")

        df = df.dropna(subset=["target"])
        logger.info("Valid samples with labels: %d", len(df))
        logger.info("Target stats: mean=%.6f, std=%.6f", df["target"].mean(), df["target"].std())

        return df

    def merge_market_news(
        self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        tolerance_hours: int = 6
    ) -> pd.DataFrame:
        """Merge market features with news signals using as-of join."""
        if news_df is None or news_df.empty:
            logger.info("No news data - using market features only")
            return market_df

        logger.info("Merging market features with news signals...")

        merged_rows = []
        tolerance = pd.Timedelta(hours=tolerance_hours)

        market_df = market_df.sort_values('time')
        news_df = news_df.sort_values('signal_time')

        news_features = [
            'signal_time', 'avg_sentiment', 'signal_strength',
            'trading_signal', 'article_count', 'quality_score'
        ]
        available_news = [c for c in news_features if c in news_df.columns]

        for _, market_row in market_df.iterrows():
            market_time = market_row['time']
            news_cutoff = market_time - tolerance
            eligible_news = news_df[
                (news_df['signal_time'] <= market_time) &
                (news_df['signal_time'] >= news_cutoff)
            ]

            merged_row = market_row.to_dict()

            if not eligible_news.empty:
                latest_news = eligible_news.iloc[-1]
                for col in available_news:
                    if col != 'signal_time':
                        merged_row[f'news_{col}'] = latest_news[col]
                news_age_minutes = (market_time - latest_news['signal_time']).total_seconds() / 60
                merged_row['news_age_minutes'] = news_age_minutes
                merged_row['news_available'] = 1
            else:
                for col in available_news:
                    if col != 'signal_time':
                        merged_row[f'news_{col}'] = 0.0
                merged_row['news_age_minutes'] = np.nan
                merged_row['news_available'] = 0

            merged_rows.append(merged_row)

        combined_df = pd.DataFrame(merged_rows)

        logger.info(f"Merged dataset: {len(combined_df)} observations")
        news_coverage = combined_df['news_available'].mean()
        logger.info(f"News coverage: {news_coverage:.1%}")

        return combined_df
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, list]:
        """Prepare market AND news features and create lagged exogenous variables."""
        logger.info("Preparing lagged market + news features for ARIMAX...")

        df = df.sort_values("time").reset_index(drop=True)

        exclude_cols = {
            "time",
            "instrument",
            "granularity",
            "close",
            "target",
            "target_classification",
            "target_regression",
            "target_pct_change",
            "target_multiclass",
            "future_close",
            "signal_time",
            "collected_at",
            "event_timestamp",
            "prediction_horizon_minutes",
            "label_generated_at",
            "fold"
        }

        # INCLUDE news features - removed the "not c.startswith("news_")" filter
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy()
        y = df["target"].astype(float).copy()
        time_index = df["time"].copy()

        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
        for col in categorical_cols:
            X[col] = X[col].astype("category").cat.codes

        X = X.apply(pd.to_numeric, errors="coerce")

        # Forward-fill missing values to avoid using future information
        X = X.ffill()

        # Remove constant or all-NaN columns
        constant_cols = X.columns[X.nunique() <= 1]
        if len(constant_cols) > 0:
            logger.info("Removing %d constant columns", len(constant_cols))
            X = X.drop(columns=list(constant_cols))
        X = X.dropna(axis=1, how="all")

        lagged_frames = [X]
        for lag in range(1, self.max_lag + 1):
            lagged = X.shift(lag).add_suffix(f"_lag{lag}")
            lagged_frames.append(lagged)

        X_lagged = pd.concat(lagged_frames, axis=1).dropna().astype(float)

        y_aligned = y.loc[X_lagged.index]
        time_aligned = time_index.loc[X_lagged.index]

        logger.info("Final feature set: %d features, %d samples", X_lagged.shape[1], X_lagged.shape[0])

        return X_lagged, y_aligned, time_aligned, list(X_lagged.columns)
    def split_train_val_test_oot(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_index: pd.Series,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        oot_ratio: float = 0.10
    ) -> Dict:
        """Create time-based splits for Train/Val/Test/OOT."""
        logger.info("\n" + "=" * 80)
        logger.info("TEMPORAL DATA SPLITTING")
        logger.info("=" * 80)

        sorted_idx = time_index.sort_values().index
        X_sorted = X.loc[sorted_idx]
        time_sorted = time_index.loc[sorted_idx]

        n = len(X_sorted)
        if n == 0:
            raise ValueError("No samples available after preprocessing.")

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        test_end = int(n * (train_ratio + val_ratio + test_ratio))

        splits = {
            "train_idx": X_sorted.index[:train_end],
            "val_idx": X_sorted.index[train_end:val_end],
            "test_idx": X_sorted.index[val_end:test_end],
            "oot_idx": X_sorted.index[test_end:],
            "train_dates": (
                time_sorted.iloc[0],
                time_sorted.iloc[train_end - 1] if train_end > 0 else time_sorted.iloc[0]
            ),
            "val_dates": (
                time_sorted.iloc[train_end] if train_end < n else time_sorted.iloc[-1],
                time_sorted.iloc[val_end - 1] if val_end > train_end else time_sorted.iloc[-1]
            ),
            "test_dates": (
                time_sorted.iloc[val_end] if val_end < n else time_sorted.iloc[-1],
                time_sorted.iloc[test_end - 1] if test_end > val_end else time_sorted.iloc[-1]
            ),
            "oot_dates": (
                time_sorted.iloc[test_end] if test_end < n else time_sorted.iloc[-1],
                time_sorted.iloc[-1]
            )
        }

        logger.info(
            "Train: %d samples (%.0f%%) | %s to %s",
            len(splits["train_idx"]),
            train_ratio * 100,
            splits["train_dates"][0],
            splits["train_dates"][1]
        )
        logger.info(
            "Val:   %d samples (%.0f%%) | %s to %s",
            len(splits["val_idx"]),
            val_ratio * 100,
            splits["val_dates"][0],
            splits["val_dates"][1]
        )
        logger.info(
            "Test:  %d samples (%.0f%%) | %s to %s",
            len(splits["test_idx"]),
            test_ratio * 100,
            splits["test_dates"][0],
            splits["test_dates"][1]
        )
        logger.info(
            "OOT:   %d samples (%.0f%%) | %s to %s",
            len(splits["oot_idx"]),
            oot_ratio * 100,
            splits["oot_dates"][0],
            splits["oot_dates"][1]
        )
        logger.info("=" * 80 + "\n")

        return splits

    def _determine_stationary_order(self, y: pd.Series) -> int:
        """Determine differencing order to achieve stationarity using ADF test."""
        best_p_value = None
        best_d = 0

        for d in range(self.max_d + 1):
            if d == 0:
                diff_series = y
            else:
                diff_series = y.diff(d).dropna()

            if len(diff_series) < 10:
                continue

            adf_stat, p_value, _, _, _, _ = adfuller(diff_series, autolag="AIC")
            logger.info("ADF test for d=%d returned p-value %.6f", d, p_value)

            if best_p_value is None or p_value < best_p_value:
                best_p_value = p_value
                best_d = d

            if p_value <= self.stationarity_alpha:
                logger.info("Selected differencing order d=%d to ensure stationarity", d)
                return d

        if best_p_value is None:
            best_p_value = float('nan')
        raise ValueError(
            f"Unable to achieve stationarity within differencing range 0-{self.max_d}. "
            f"Best p-value obtained was {best_p_value:.6f} at d={best_d}."
        )

    @staticmethod
    def _get_exog_matrix(X: pd.DataFrame):
        """Return numpy array for exogenous features or None if not available."""
        if X is None or X.empty or X.shape[1] == 0:
            return None
        return X.to_numpy()

    @staticmethod
    def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
        """Compute evaluation metrics for predictions."""
        metrics = {}
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        metrics[f"{prefix}_rmse"] = rmse
        metrics[f"{prefix}_mae"] = mae

        mask = np.abs(y_true) > 1e-8
        if mask.any():
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            metrics[f"{prefix}_mape"] = mape

        return metrics
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_index: pd.Series,
        feature_names: list
    ) -> Tuple[object, Dict]:
        """Train Auto-ARIMA model with proper train/val/test/OOT splits and MLflow tracking."""
        logger.info("Training Auto-ARIMA model with MLflow tracking...")

        splits = self.split_train_val_test_oot(X, y, time_index)

        X_train = X.loc[splits["train_idx"]]
        y_train = y.loc[splits["train_idx"]]

        X_val = X.loc[splits["val_idx"]]
        y_val = y.loc[splits["val_idx"]]

        X_test = X.loc[splits["test_idx"]]
        y_test = y.loc[splits["test_idx"]]

        X_oot = X.loc[splits["oot_idx"]]
        y_oot = y.loc[splits["oot_idx"]]

        d = self._determine_stationary_order(y_train)

        metrics: Dict[str, float] = {}

        with mlflow.start_run():
            mlflow.log_param("prediction_horizon_minutes", self.prediction_horizon)
            mlflow.log_param("task_type", self.task)
            mlflow.log_param("n_total_samples", len(X))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_oot_samples", len(X_oot))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("news_available", 'news_available' in X.columns)
            mlflow.log_param("max_lag", self.max_lag)
            mlflow.log_param("max_p", self.max_p)
            mlflow.log_param("max_q", self.max_q)
            mlflow.log_param("stationarity_alpha", self.stationarity_alpha)
            mlflow.log_param("selected_d", d)

            exog_train = self._get_exog_matrix(X_train)

            logger.info("Fitting auto_arima with d=%d (max p=%d, max q=%d)...", d, self.max_p, self.max_q)
            model = auto_arima(
                y_train.to_numpy(),
                exogenous=exog_train,
                d=d,
                start_p=0,
                start_q=0,
                max_p=self.max_p,
                max_q=self.max_q,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="raise",
                with_intercept=True
            )

            order = model.order
            mlflow.log_param("order_p", order[0])
            mlflow.log_param("order_d", order[1])
            mlflow.log_param("order_q", order[2])
            mlflow.log_metric("train_aic", float(model.aic()))
            mlflow.set_tag("model_type", "auto_arima")

            train_pred = model.predict_in_sample(exogenous=exog_train)
            metrics.update(self._evaluate_predictions(y_train.to_numpy(), train_pred, prefix="train"))

            if len(y_val) > 0:
                exog_val = self._get_exog_matrix(X_val)
                val_pred = model.predict(n_periods=len(y_val), exogenous=exog_val)
                metrics.update(self._evaluate_predictions(y_val.to_numpy(), val_pred, prefix="val"))
                model.update(y_val.to_numpy(), exog_val)

            if len(y_test) > 0:
                exog_test = self._get_exog_matrix(X_test)
                test_pred = model.predict(n_periods=len(y_test), exogenous=exog_test)
                metrics.update(self._evaluate_predictions(y_test.to_numpy(), test_pred, prefix="test"))
                model.update(y_test.to_numpy(), exog_test)

            if len(y_oot) > 0:
                exog_oot = self._get_exog_matrix(X_oot)
                oot_pred = model.predict(n_periods=len(y_oot), exogenous=exog_oot)
                metrics.update(self._evaluate_predictions(y_oot.to_numpy(), oot_pred, prefix="oot"))

            for name, value in metrics.items():
                if value is not None and not np.isnan(value):
                    mlflow.log_metric(name, float(value))

        return model, metrics

    def save_model_and_artifacts(
        self,
        model,
        feature_names: list,
        metrics: Dict
    ):
        """Save trained model and artifacts locally."""
        logger.info("Saving model artifacts locally...")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"arima_{self.prediction_horizon}min_{timestamp}"

        model_path = self.output_dir / f"{model_name}.pkl"
        dump(model, model_path)
        logger.info("  Model saved: %s", model_path)

        feature_path = self.output_dir / f"{model_name}_features.json"
        with open(feature_path, "w") as f:
            json.dump({"features": feature_names}, f, indent=2)
        logger.info("  Features saved: %s", feature_path)

        clean_metrics = {k: float(v) for k, v in metrics.items() if v is not None and not np.isnan(v)}
        metrics_path = self.output_dir / f"{model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(clean_metrics, f, indent=2)
        logger.info("  Metrics saved: %s", metrics_path)

    def run(self):
        """Execute the full training pipeline with MLflow."""
        logger.info("\n" + "=" * 80)
        logger.info("ARIMAX Training Pipeline with MLflow - %smin Prediction", self.prediction_horizon)
        logger.info("=" * 80 + "\n")

        # Load data (market features + gold labels + news signals)
        market_df, news_df = self.load_data()

        # Validate labels are present
        market_df = self.validate_labels(market_df)

        # Merge with news
        combined_df = self.merge_market_news(market_df, news_df)

        X, y, time_index, feature_names = self.prepare_features(combined_df)

        model, metrics = self.train_model(X, y, time_index, feature_names)

        self.save_model_and_artifacts(model, feature_names, metrics)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info("Prediction horizon: %s minutes", self.prediction_horizon)
        logger.info("Total samples: %d", len(X))
        logger.info("Features: %d", len(feature_names))
        logger.info(
            "Test Set - RMSE: %.4f, MAE: %.4f",
            metrics.get("test_rmse", float("nan")),
            metrics.get("test_mae", float("nan"))
        )
        logger.info(
            "OOT Set  - RMSE: %.4f, MAE: %.4f",
            metrics.get("oot_rmse", float("nan")),
            metrics.get("oot_mae", float("nan"))
        )
        logger.info("Model saved to: %s", self.output_dir)
        logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
        logger.info("=" * 80 + "\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-features",
        type=Path,
        required=True,
        help="Path to market features CSV/Parquet (Gold layer)"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Path to labels CSV (Gold layer). If not provided, will be inferred from market features path."
    )
    parser.add_argument(
        "--news-signals",
        type=Path,
        help="Path to news signals CSV/Parquet (Gold layer)"
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=30,
        help="Prediction horizon in minutes (default: 30)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_clean/models"),
        help="Output directory for models"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sp500_prediction_arimax",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=3,
        help="Maximum lag order to create for exogenous features"
    )
    parser.add_argument(
        "--max-p",
        type=int,
        default=5,
        help="Maximum AR order to explore in auto ARIMA"
    )
    parser.add_argument(
        "--max-q",
        type=int,
        default=5,
        help="Maximum MA order to explore in auto ARIMA"
    )
    parser.add_argument(
        "--max-d",
        type=int,
        default=2,
        help="Maximum differencing order to consider when enforcing stationarity"
    )
    parser.add_argument(
        "--stationarity-alpha",
        type=float,
        default=0.05,
        help="Significance level for the ADF stationarity test"
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_registry_uri(args.mlflow_uri)

    pipeline = ARIMAMLflowTrainingPipeline(
        market_features_path=args.market_features,
        labels_path=args.labels,
        news_signals_path=args.news_signals,
        prediction_horizon_minutes=args.prediction_horizon,
        output_dir=args.output_dir,
        task="regression",
        experiment_name=args.experiment_name,
        max_lag=args.max_lag,
        max_p=args.max_p,
        max_q=args.max_q,
        max_d=args.max_d,
        stationarity_alpha=args.stationarity_alpha
    )

    pipeline.run()


if __name__ == "__main__":
    main()
