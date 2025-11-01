"""ARIMA-based inference implementation."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .inference import ModelInference

logger = logging.getLogger(__name__)


class ModelInferenceARIMA(ModelInference):
    """Specialized inference engine backed by a statsmodels ARIMA model."""

    def __init__(
        self,
        model_path: str = "models/arima_model.pkl",
        feast_repo: str = "feature_repo",
    ):
        """Initialize ARIMA inference engine."""
        super().__init__(model_path=model_path, feast_repo=feast_repo)

    def _load_model(self) -> None:
        """Load a statsmodels ARIMA model bundle."""
        try:
            from statsmodels.tsa.arima.model import ARIMAResults
        except ImportError as exc:
            logger.error("statsmodels is required for ARIMA inference: %s", exc)
            self.is_loaded = False
            return

        if not self.model_path.exists():
            logger.warning("ARIMA model not found at %s", self.model_path)
            self.is_loaded = False
            return

        try:
            self.model = ARIMAResults.load(str(self.model_path))
            self.model_type = "arima"
            self.scaler = None
            self.feature_names = ["close"]
            self.is_loaded = True
            logger.info(
                "Loaded ARIMA model with %d original observations",
                getattr(self.model, "nobs", -1),
            )
        except Exception as exc:
            logger.error("Error loading ARIMA model: %s", exc)
            self.is_loaded = False

    def get_online_features(self, instrument: str) -> Optional[Dict]:
        """Fetch online features from Feast.

        Args:
            instrument: Trading instrument (e.g., SPX500_USD)

        Returns:
            Dictionary of features or None if unavailable
        """
        if not self.feast_store:
            return None

        try:
            entity_rows = [{"instrument": instrument}]

            # Fetch both market and news features
            features_to_fetch = [
                "market_gold_features:ret_30",
            ]

            online_features = self.feast_store.get_online_features(
                features=features_to_fetch,
                entity_rows=entity_rows
            ).to_dict()

            return online_features

        except Exception as e:
            logger.warning(f"Could not fetch Feast features: {e}")
            return None


    def _prepare_features(
        self, features: Union[pd.DataFrame, Dict]
    ) -> pd.DataFrame:
        """Create lagged features and returns for ARIMA forecasting."""
        if isinstance(features, pd.DataFrame):
            X = features.copy()
        else:
            X = pd.DataFrame(features)

        if X.empty:
            raise ValueError("No features provided for ARIMA inference")

        required_columns = {"time", "close"}
        missing = required_columns - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for ARIMA inference: {missing}"
            )

        return X

    def _predict(
        self,
        instrument: str,
        prepared_features: pd.DataFrame,
    ) -> Dict:
        """Run ARIMA forecast against preprocessed features."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("ARIMA model not loaded")

        prepared_features = prepared_features.sort_values("time")
        close_series = prepared_features["close"].astype(float)

        try:
            # Update the ARIMA state with the latest observations without refitting.
            updated_model = self.model.apply(close_series, refit=False)
        except Exception as exc:
            logger.debug("Falling back to stored ARIMA results: %s", exc)
            updated_model = self.model

        forecast_steps = 1
        forecast = updated_model.forecast(steps=forecast_steps)
        forecast_array = np.asarray(forecast, dtype=float)
        if forecast_array.size == 0:
            raise ValueError("ARIMA forecast returned no values")

        predicted_close = float(forecast_array[-1])
        last_close = float(close_series.iloc[-1])
        last_time = pd.to_datetime(prepared_features["time"].iloc[-1])
        forecast_time = (last_time + timedelta(minutes=1)).to_pydatetime()

        price_delta = predicted_close - last_close
        scale = max(abs(last_close), 1e-6)
        signal_strength = float(np.tanh(price_delta / scale))
        probability = float(np.clip(0.5 + signal_strength / 2.0, 0.0, 1.0))
        confidence = abs(probability - 0.5) * 2.0

        return {
            "instrument": instrument,
            "timestamp": forecast_time,
            "prediction": "bullish" if price_delta >= 0 else "bearish",
            "probability": probability,
            "confidence": float(confidence),
            "signal_strength": signal_strength,
            "predicted_close": predicted_close,
            "last_close": last_close,
            "model_version": self.model_type or "arima",
        }

    def predict(
        self,
        instrument: str = "SPX500_USD",
        timestamp: Optional[datetime] = None,
        historical_data: Optional[Union[pd.DataFrame, Dict]] = None,
    ) -> Dict:
        """Generate ARIMA-based prediction using historical price data."""
        timestamp = timestamp or datetime.now()

        if not self.is_loaded:
            logger.warning("ARIMA model unavailable - returning mock prediction")
            return self._mock_prediction(instrument, timestamp)

        if historical_data is None:
            logger.warning("Historical data required for ARIMA inference")
            return self._mock_prediction(instrument, timestamp)

        try:
            prepared = self._prepare_features(historical_data)
            return self._predict(instrument, prepared)
        except Exception as exc:
            logger.error("Error during ARIMA prediction: %s", exc)
            return self._mock_prediction(instrument, timestamp)
