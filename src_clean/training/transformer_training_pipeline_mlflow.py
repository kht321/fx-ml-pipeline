"""
Transformer Training Pipeline with MLflow Integration - 30-Minute Price Prediction

Financial Transformer model for time series prediction with:
- Temporal attention mechanism
- Proper time-based train/val/test/OOT splits
- MLflow experiment tracking
- GPU support (falls back to CPU)

Based on Gabriel's implementation with production enhancements.

Repository Location: fx-ml-pipeline/src_clean/training/transformer_training_pipeline_mlflow.py
"""

import argparse
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import mlflow
import mlflow.pytorch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransformerConfig:
    """Configuration for Financial Transformer model."""
    # Model architecture
    D_MODEL = 512
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 2048
    DROPOUT = 0.2
    SEQUENCE_LENGTH = 50  # Time steps to look back

    # Training
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5


class FinancialTransformer(nn.Module):
    """Transformer model for financial time series prediction."""

    def __init__(self, config: TransformerConfig, n_features: int):
        super().__init__()
        self.config = config
        self.n_features = n_features

        # Input embedding for financial features
        self.feature_embedding = nn.Linear(n_features, config.D_MODEL)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(config.SEQUENCE_LENGTH, config.D_MODEL)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=config.D_FF,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.N_LAYERS)

        # Output head for binary classification
        self.output_head = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MODEL // 2, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape

        # Feature embedding
        x = self.feature_embedding(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)

        # Transformer
        transformer_out = self.transformer(x)  # (batch_size, seq_len, d_model)

        # Use last hidden state
        last_hidden = transformer_out[:, -1, :]  # (batch_size, d_model)

        # Output
        output = self.output_head(last_hidden)  # (batch_size, 1)

        return output


class TransformerMLflowTrainingPipeline:
    """Training pipeline for Financial Transformer with MLflow tracking."""

    def __init__(
        self,
        market_features_path: Path,
        labels_path: Optional[Path] = None,
        news_signals_path: Optional[Path] = None,
        prediction_horizon_minutes: int = 30,
        output_dir: Path = Path("data_clean/models"),
        task: str = "classification",
        experiment_name: str = "sp500_prediction_transformer",
        config: Optional[TransformerConfig] = None
    ):
        """Initialize transformer training pipeline."""
        self.market_features_path = market_features_path
        self.labels_path = labels_path
        self.news_signals_path = news_signals_path
        self.prediction_horizon = prediction_horizon_minutes
        self.output_dir = output_dir
        self.task = task
        self.experiment_name = experiment_name
        self.config = config or TransformerConfig()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

        self.scaler = StandardScaler()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load market features, labels, and news signals from Gold layer."""
        logger.info("Loading data from Gold layer...")

        # Load market features
        market_df = pd.read_csv(self.market_features_path)
        market_df['time'] = pd.to_datetime(market_df['time'])
        logger.info(f"Loaded market features: {len(market_df)} rows, {len(market_df.columns)} columns")

        # Load or infer labels path
        if self.labels_path is None:
            features_dir = self.market_features_path.parent
            base_dir = features_dir.parent.parent.parent
            instrument = self.market_features_path.stem.replace('_features', '')
            self.labels_path = base_dir / f"gold/market/labels/{instrument}_labels_{self.prediction_horizon}min.csv"
            logger.info(f"Inferred labels path: {self.labels_path}")

        # Load labels
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        labels_df = pd.read_csv(self.labels_path)
        labels_df['time'] = pd.to_datetime(labels_df['time'])
        logger.info(f"Loaded gold labels: {len(labels_df)} rows, {len(labels_df.columns)} columns")

        # Merge market features with labels
        logger.info("Merging market features with gold labels...")
        merged_df = pd.merge(
            market_df,
            labels_df[['time', 'target_classification', 'target_regression', 'target_pct_change', 'fold']],
            on='time',
            how='inner'
        )
        logger.info(f"Merged dataset: {len(merged_df)} rows after merging")

        # Set target
        if self.task == "classification":
            merged_df['target'] = merged_df['target_classification']
        else:
            merged_df['target'] = merged_df['target_pct_change']

        # Load news signals if available
        news_df = None
        if self.news_signals_path and self.news_signals_path.exists():
            news_df = pd.read_csv(self.news_signals_path)
            news_df['signal_time'] = pd.to_datetime(news_df['signal_time'])
            logger.info(f"Loaded news signals: {len(news_df)} rows, {len(news_df.columns)} columns")
        else:
            logger.warning("No news signals provided - using market features only")

        return merged_df, news_df

    def merge_market_news(
        self,
        market_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        tolerance_hours: int = 6
    ) -> pd.DataFrame:
        """Merge market features with news signals using as-of join."""
        if news_df is None or news_df.empty:
            logger.info("No news data - using market features only")
            # Add dummy news columns with zeros
            market_df['news_avg_sentiment'] = 0.0
            market_df['news_signal_strength'] = 0.0
            market_df['news_trading_signal'] = 0.0
            market_df['news_article_count'] = 0.0
            market_df['news_quality_score'] = 0.0
            market_df['news_age_minutes'] = 0.0
            market_df['news_available'] = 0
            return market_df

        logger.info("Merging market features with news signals (this may take a while)...")

        # Use merge_asof for efficient time-based merge
        market_df = market_df.sort_values('time')
        news_df = news_df.sort_values('signal_time')

        # Rename news columns to avoid conflicts
        news_df_renamed = news_df.rename(columns={'signal_time': 'time'})

        # Perform merge_asof (forward fill news data)
        merged_df = pd.merge_asof(
            market_df,
            news_df_renamed[['time', 'avg_sentiment', 'signal_strength', 'trading_signal',
                            'article_count', 'quality_score']],
            on='time',
            direction='backward',
            tolerance=pd.Timedelta(hours=tolerance_hours),
            suffixes=('', '_news')
        )

        # Rename and fill missing values
        news_cols_map = {
            'avg_sentiment': 'news_avg_sentiment',
            'signal_strength': 'news_signal_strength',
            'trading_signal': 'news_trading_signal',
            'article_count': 'news_article_count',
            'quality_score': 'news_quality_score'
        }

        for old_col, new_col in news_cols_map.items():
            if old_col in merged_df.columns:
                merged_df[new_col] = merged_df[old_col].fillna(0.0)
                merged_df.drop(old_col, axis=1, inplace=True)
            else:
                merged_df[new_col] = 0.0

        merged_df['news_available'] = (merged_df['news_avg_sentiment'] != 0).astype(int)
        merged_df['news_age_minutes'] = 0.0  # Simplified for merge_asof

        news_coverage = merged_df['news_available'].mean()
        logger.info(f"News coverage: {news_coverage:.1%}")

        return merged_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, list]:
        """Prepare features for training."""
        logger.info("Preparing features...")

        # Exclude label-related and metadata columns
        exclude_cols = {
            'time', 'instrument', 'granularity', 'close',
            'target', 'target_classification', 'target_regression',
            'target_pct_change', 'target_multiclass', 'future_close',
            'signal_time', 'collected_at', 'event_timestamp',
            'prediction_horizon_minutes', 'label_generated_at', 'fold'
        }

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy()
        y = df['target'].copy()
        time_index = df['time'].copy()

        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes

        # Fill missing values
        X = X.fillna(X.median())

        # Remove constant columns
        constant_cols = X.columns[X.nunique() <= 1]
        if len(constant_cols) > 0:
            logger.info(f"Removing {len(constant_cols)} constant columns")
            X = X.drop(constant_cols, axis=1)

        logger.info(f"Final feature set: {len(X.columns)} features, {len(X)} samples")

        return X, y, time_index, list(X.columns)

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
        logger.info("\n" + "="*80)
        logger.info("TEMPORAL DATA SPLITTING")
        logger.info("="*80)

        # Sort by time
        sorted_idx = time_index.sort_values().index
        X_sorted = X.loc[sorted_idx]
        y_sorted = y.loc[sorted_idx]
        time_sorted = time_index.loc[sorted_idx]

        n = len(X_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        test_end = int(n * (train_ratio + val_ratio + test_ratio))

        splits = {
            'train_idx': X_sorted.index[:train_end],
            'val_idx': X_sorted.index[train_end:val_end],
            'test_idx': X_sorted.index[val_end:test_end],
            'oot_idx': X_sorted.index[test_end:],
            'train_dates': (time_sorted.iloc[0], time_sorted.iloc[train_end-1]),
            'val_dates': (time_sorted.iloc[train_end], time_sorted.iloc[val_end-1]),
            'test_dates': (time_sorted.iloc[val_end], time_sorted.iloc[test_end-1]),
            'oot_dates': (time_sorted.iloc[test_end], time_sorted.iloc[-1])
        }

        # Log split information
        logger.info(f"Total samples: {n}")
        logger.info(f"Train: {len(splits['train_idx'])} samples ({train_ratio*100:.0f}%) | "
                   f"{splits['train_dates'][0]} to {splits['train_dates'][1]}")
        logger.info(f"Val:   {len(splits['val_idx'])} samples ({val_ratio*100:.0f}%) | "
                   f"{splits['val_dates'][0]} to {splits['val_dates'][1]}")
        logger.info(f"Test:  {len(splits['test_idx'])} samples ({test_ratio*100:.0f}%) | "
                   f"{splits['test_dates'][0]} to {splits['test_dates'][1]}")
        logger.info(f"OOT:   {len(splits['oot_idx'])} samples ({oot_ratio*100:.0f}%) | "
                   f"{splits['oot_dates'][0]} to {splits['oot_dates'][1]}")
        logger.info("="*80 + "\n")

        return splits

    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences for transformer input."""
        num_sequences = len(features) - self.config.SEQUENCE_LENGTH + 1

        if num_sequences <= 0:
            raise ValueError(f"Not enough data for sequence length {self.config.SEQUENCE_LENGTH}")

        sequences = np.zeros((num_sequences, self.config.SEQUENCE_LENGTH, features.shape[1]), dtype=np.float32)
        sequence_targets = np.zeros(num_sequences, dtype=np.float32)

        for i in range(num_sequences):
            sequences[i] = features[i:i + self.config.SEQUENCE_LENGTH]
            sequence_targets[i] = targets[i + self.config.SEQUENCE_LENGTH - 1]

        return torch.from_numpy(sequences).float(), torch.from_numpy(sequence_targets).float()

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_index: pd.Series,
        feature_names: list
    ) -> Tuple[FinancialTransformer, Dict]:
        """Train transformer model with MLflow tracking."""
        logger.info("Training Financial Transformer model...")

        # Create temporal splits
        splits = self.split_train_val_test_oot(X, y, time_index)

        # Extract split data
        X_train = X.loc[splits['train_idx']]
        y_train = y.loc[splits['train_idx']]

        X_val = X.loc[splits['val_idx']]
        y_val = y.loc[splits['val_idx']]

        X_test = X.loc[splits['test_idx']]
        y_test = y.loc[splits['test_idx']]

        X_oot = X.loc[splits['oot_idx']]
        y_oot = y.loc[splits['oot_idx']]

        # Fit scaler on training data only
        self.scaler.fit(X_train)

        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        X_oot_scaled = self.scaler.transform(X_oot)

        # Create sequences
        logger.info("Creating sequences...")
        train_seq, train_targets = self.create_sequences(X_train_scaled, y_train.values)
        val_seq, val_targets = self.create_sequences(X_val_scaled, y_val.values)
        test_seq, test_targets = self.create_sequences(X_test_scaled, y_test.values)
        oot_seq, oot_targets = self.create_sequences(X_oot_scaled, y_oot.values)

        logger.info(f"Train sequences: {len(train_seq)}, Val sequences: {len(val_seq)}")
        logger.info(f"Test sequences: {len(test_seq)}, OOT sequences: {len(oot_seq)}")

        # Create data loaders
        train_dataset = TensorDataset(train_seq, train_targets)
        val_dataset = TensorDataset(val_seq, val_targets)
        test_dataset = TensorDataset(test_seq, test_targets)
        oot_dataset = TensorDataset(oot_seq, oot_targets)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        oot_loader = DataLoader(
            oot_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", "transformer")
            mlflow.log_param("prediction_horizon_minutes", self.prediction_horizon)
            mlflow.log_param("task_type", self.task)
            mlflow.log_param("n_total_samples", len(X))
            mlflow.log_param("n_train_sequences", len(train_seq))
            mlflow.log_param("n_val_sequences", len(val_seq))
            mlflow.log_param("n_test_sequences", len(test_seq))
            mlflow.log_param("n_oot_sequences", len(oot_seq))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("sequence_length", self.config.SEQUENCE_LENGTH)
            mlflow.log_param("d_model", self.config.D_MODEL)
            mlflow.log_param("n_heads", self.config.N_HEADS)
            mlflow.log_param("n_layers", self.config.N_LAYERS)
            mlflow.log_param("batch_size", self.config.BATCH_SIZE)
            mlflow.log_param("learning_rate", self.config.LEARNING_RATE)
            mlflow.log_param("news_available", 'news_available' in X.columns)

            # Initialize model
            model = FinancialTransformer(self.config, n_features=len(feature_names)).to(self.device)
            logger.info(f"Model parameters on {self.device}: {next(model.parameters()).is_cuda if torch.cuda.is_available() else 'CPU'}")

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.MAX_EPOCHS):
                # Train
                model.train()
                train_loss = 0.0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device).unsqueeze(1)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # Validate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device).unsqueeze(1)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

                # Log metrics
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_path = self.output_dir / "transformer_best_model.pth"
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1

                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Load best model
            model.load_state_dict(torch.load(best_model_path))

            # Evaluate on all splits
            logger.info("\n" + "="*80)
            logger.info("EVALUATION ON ALL SPLITS")
            logger.info("="*80)

            train_metrics = self._evaluate_model(model, train_loader, prefix="train")
            logger.info(f"Train: Accuracy={train_metrics['train_accuracy']:.4f}, AUC={train_metrics['train_auc']:.4f}")

            val_metrics = self._evaluate_model(model, val_loader, prefix="val")
            logger.info(f"Val:   Accuracy={val_metrics['val_accuracy']:.4f}, AUC={val_metrics['val_auc']:.4f}")

            test_metrics = self._evaluate_model(model, test_loader, prefix="test")
            logger.info(f"Test:  Accuracy={test_metrics['test_accuracy']:.4f}, AUC={test_metrics['test_auc']:.4f}")

            oot_metrics = self._evaluate_model(model, oot_loader, prefix="oot")
            logger.info(f"OOT:   Accuracy={oot_metrics['oot_accuracy']:.4f}, AUC={oot_metrics['oot_auc']:.4f}")

            # Aggregate metrics
            metrics = {
                **train_metrics,
                **val_metrics,
                **test_metrics,
                **oot_metrics
            }

            # Log all metrics to MLflow
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Log model
            mlflow.pytorch.log_model(model, "model")

            # Log feature names
            mlflow.log_dict({"features": feature_names}, "features.json")

            logger.info(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
            logger.info("="*80 + "\n")

            return model, metrics

    def _evaluate_model(self, model: FinancialTransformer, data_loader: DataLoader, prefix: str = "") -> Dict:
        """Evaluate model and return metrics."""
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = model(batch_x)
                predictions = torch.sigmoid(outputs).cpu().numpy()
                targets = batch_y.cpu().numpy()

                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets.flatten())

        predictions_binary = (np.array(all_predictions) > 0.5).astype(int)

        metrics = {
            f'{prefix}_accuracy' if prefix else 'accuracy': float(accuracy_score(all_targets, predictions_binary)),
            f'{prefix}_auc' if prefix else 'auc': float(roc_auc_score(all_targets, all_predictions))
        }

        if prefix in ['test', 'oot', '']:
            metrics[f'{prefix}_confusion_matrix' if prefix else 'confusion_matrix'] = confusion_matrix(all_targets, predictions_binary).tolist()

        return metrics

    def save_model_and_artifacts(
        self,
        model: FinancialTransformer,
        feature_names: list,
        metrics: Dict
    ):
        """Save trained model."""
        logger.info("Saving model artifacts locally...")

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"transformer_{self.task}_{self.prediction_horizon}min_{timestamp}"

        # Save model
        model_path = self.output_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"  Model saved: {model_path}")

        # Save scaler
        scaler_path = self.output_dir / f"{model_name}_scaler.pkl"
        dump(self.scaler, scaler_path)

        # Save feature names
        feature_path = self.output_dir / f"{model_name}_features.json"
        with open(feature_path, 'w') as f:
            json.dump({'features': feature_names}, f, indent=2)

        # Save metrics
        metrics_path = self.output_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"  Metrics saved: {metrics_path}")

    def run(self):
        """Execute the full training pipeline."""
        logger.info("\n" + "="*80)
        logger.info(f"Transformer Training Pipeline - {self.prediction_horizon}min Prediction")
        logger.info("="*80 + "\n")

        # Load data
        market_df, news_df = self.load_data()

        # Merge with news
        combined_df = self.merge_market_news(market_df, news_df)

        # Prepare features
        X, y, time_index, feature_names = self.prepare_features(combined_df)

        # Train model
        model, metrics = self.train_model(X, y, time_index, feature_names)

        # Save everything
        self.save_model_and_artifacts(model, feature_names, metrics)

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Model: Financial Transformer")
        logger.info(f"Task: {self.task}")
        logger.info(f"Prediction horizon: {self.prediction_horizon} minutes")
        logger.info(f"Features: {len(feature_names)}")
        logger.info(f"Sequence length: {self.config.SEQUENCE_LENGTH}")
        logger.info(f"\nTest Set - Accuracy: {metrics.get('test_accuracy', 0):.4f}, "
                   f"AUC: {metrics.get('test_auc', 0):.4f}")
        logger.info(f"OOT Set  - Accuracy: {metrics.get('oot_accuracy', 0):.4f}, "
                   f"AUC: {metrics.get('oot_auc', 0):.4f}")
        logger.info(f"\nModel saved to: {self.output_dir}")
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        logger.info("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market-features",
        type=Path,
        required=True,
        help="Path to market features CSV (Gold layer)"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Path to labels CSV (Gold layer)"
    )
    parser.add_argument(
        "--news-signals",
        type=Path,
        help="Path to news signals CSV (Gold layer)"
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=30,
        help="Prediction horizon in minutes (default: 30)"
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Prediction task type"
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
        default="sp500_prediction_transformer",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI"
    )

    args = parser.parse_args()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_uri)

    pipeline = TransformerMLflowTrainingPipeline(
        market_features_path=args.market_features,
        labels_path=args.labels,
        news_signals_path=args.news_signals,
        prediction_horizon_minutes=args.prediction_horizon,
        output_dir=args.output_dir,
        task=args.task,
        experiment_name=args.experiment_name
    )

    pipeline.run()


if __name__ == "__main__":
    main()
