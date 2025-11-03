"""Model inference engine with Feast integration."""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Prediction logging directory
PREDICTIONS_LOG_DIR = Path("data_clean/predictions")
PREDICTIONS_LOG_FILE = PREDICTIONS_LOG_DIR / "prediction_log.jsonl"


class ModelInference:
    """Handles ML model inference with feature fetching."""

    def __init__(
        self,
        model_path: str = "models/gradient_boosting_combined_model.pkl",
        feast_repo: str = "feature_repo",
        prediction_task: Optional[str] = None,
        price_feature_key: Optional[str] = None,
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to trained model pickle file
            feast_repo: Path to Feast feature repository
            prediction_task: Optional override for the prediction task ("classification" or "regression")
            price_feature_key: Optional feature key that represents the current price for regression models
        """
        self.model_path = Path(model_path)
        self.feast_repo = Path(feast_repo)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_type = None
        self.prediction_task = prediction_task
        self.price_feature_key = price_feature_key
        self.feast_store = None
        self.is_loaded = False

        self._load_model()
        self._init_feast()

    def _load_model(self):
        """Load the trained ML model."""
        try:
            from joblib import load
            import glob

            if not self.model_path.exists():
                logger.warning(f"Model not found at {self.model_path}")
                # Try alternative paths - PRIORITIZE NEWEST REGRESSION MODELS for price prediction

                # First, try to find the newest XGBoost regression model (search multiple locations)
                search_patterns = [
                    "models/xgboost/xgboost_regression_*.pkl",
                    "models/lightgbm/lightgbm_regression_*.pkl",
                    "data_clean/models/xgboost_regression_*.pkl",
                ]

                xgb_models = []
                for pattern in search_patterns:
                    found = glob.glob(pattern, recursive=False)  # Non-recursive to avoid subdirs
                    xgb_models.extend(found)

                # Filter out models with insufficient features
                valid_models = []
                for model_candidate in xgb_models:
                    try:
                        from joblib import load as joblib_load
                        test_bundle = joblib_load(model_candidate)
                        if isinstance(test_bundle, dict):
                            test_features = test_bundle.get('feature_names', [])
                        else:
                            test_features = getattr(test_bundle, 'feature_names_in_', [])

                        # Only use models with at least 50 features
                        if len(test_features) >= 50:
                            valid_models.append(model_candidate)
                    except:
                        pass

                # Sort by modification time, newest first
                valid_models = sorted(valid_models, key=lambda x: Path(x).stat().st_mtime, reverse=True)

                alternative_paths = []
                # Add valid models first
                alternative_paths.extend([Path(p) for p in valid_models[:3]])

                # Then try data_clean/models
                alternative_paths.extend([
                    Path("data_clean/models/xgboost_regression_30min_20251026_030337.pkl"),
                    Path("models/lightgbm/lightgbm_regression_30min_20251026_030405.pkl"),
                    Path("data_clean/models/xgboost_classification_30min_20251101_042201.pkl"),
                    Path("models/xgboost_combined_model.pkl"),
                    Path("data_clean/models/xgboost_classification_enhanced.pkl"),
                    Path("models/random_forest_combined_model.pkl"),
                    Path("data/combined/models/gradient_boosting_combined_model.pkl"),
                ])

                for alt_path in alternative_paths:
                    if alt_path.exists():
                        self.model_path = alt_path
                        logger.info(f"Using alternative model: {alt_path}")
                        break

            if self.model_path.exists():
                model_bundle = load(self.model_path)

                # Handle both dictionary bundles and raw model objects
                if isinstance(model_bundle, dict):
                    self.model = model_bundle.get('model')
                    self.scaler = model_bundle.get('scaler')
                    self.feature_names = model_bundle.get('feature_names', [])
                    self.model_type = model_bundle.get('model_type', 'unknown')
                    bundle_task = model_bundle.get('task_type') or model_bundle.get('prediction_task')
                    if not self.prediction_task:
                        self.prediction_task = bundle_task or self._infer_task_from_model()
                else:
                    # Raw model object (e.g., LGBMClassifier, XGBClassifier)
                    self.model = model_bundle
                    self.scaler = None
                    feature_names_raw = getattr(model_bundle, 'feature_names_in_', [])
                    # Convert numpy array to list if needed
                    self.feature_names = list(feature_names_raw) if hasattr(feature_names_raw, '__iter__') else []
                    self.model_type = type(model_bundle).__name__
                    if not self.prediction_task:
                        self.prediction_task = self._infer_task_from_model()

                self.is_loaded = True
                logger.info(f"Loaded {self.model_type} model with {len(self.feature_names)} features")
            else:
                logger.warning("No model found - predictions will be mock")
                self.is_loaded = False

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False

    def _init_feast(self):
        """Initialize Feast feature store connection."""
        try:
            from feast import FeatureStore

            if self.feast_repo.exists():
                self.feast_store = FeatureStore(repo_path=str(self.feast_repo))
                logger.info("Feast feature store initialized")
            else:
                logger.warning(f"Feast repo not found at {self.feast_repo}")

        except ImportError:
            logger.warning("Feast not installed - will use fallback feature fetching")
        except Exception as e:
            logger.error(f"Error initializing Feast: {e}")

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

            # Fetch both market and news features from our Feast feature views
            features_to_fetch = [
                "market_features:close",
                "market_features:rsi_14",
                "market_features:macd",
                "market_features:macd_signal",
                "market_features:bb_upper",
                "market_features:bb_middle",
                "market_features:bb_lower",
                "market_features:sma_7",
                "market_features:sma_14",
                "market_features:ema_7",
                "market_features:ema_14",
                "market_features:atr_14",
                "market_features:adx_14",
                "market_features:volatility_20",
                "news_signals:avg_sentiment",
                "news_signals:signal_strength",
                "news_signals:article_count",
            ]

            online_features = self.feast_store.get_online_features(
                features=features_to_fetch,
                entity_rows=entity_rows
            ).to_dict()

            return online_features

        except Exception as e:
            logger.warning(f"Could not fetch Feast features: {e}")
            return None

    def predict(self, instrument: str = "SPX500_USD", timestamp: Optional[datetime] = None) -> Dict:
        """Generate prediction for given instrument.

        Args:
            instrument: Trading instrument
            timestamp: Optional timestamp (for historical predictions)

        Returns:
            Prediction dictionary
        """
        timestamp = timestamp or datetime.now()

        if not self.is_loaded:
            # Return mock prediction if model not loaded
            return self._mock_prediction(instrument, timestamp)

        # For full 70-feature model, we need to calculate features on-the-fly
        # Feast features are supplementary (news sentiment)
        # Try to get news features from Feast
        feast_features = self.get_online_features(instrument)

        # Create feature dict with all 70 features
        # For now, use a simplified approach with mock/default values
        # In production, you'd fetch real OANDA data and calculate all features
        features = self._get_full_feature_set(instrument, feast_features)

        try:
            # Prepare feature vector
            feature_vector = self._prepare_features(features)
            feature_array = np.array([feature_vector], dtype=float)

            if self.scaler:
                feature_array = self.scaler.transform(feature_array)

            task = (self.prediction_task or self._infer_task_from_model()).lower()

            if task == "regression":
                relative_change = float(self.model.predict(feature_array)[0])

                prediction_label = self._relative_change_to_label(relative_change)
                probability = None
                confidence = min(abs(relative_change), 1.0)
                signal_strength = relative_change
                predicted_price = self._calculate_predicted_price(features, relative_change)
                predicted_relative_change = relative_change
            else:
                # Default to classification behaviour
                prediction_proba = self.model.predict_proba(feature_array)[0]
                prediction_class = self.model.predict(feature_array)[0]

                probability = float(prediction_proba[1])  # Probability of bullish

                prediction_label = "bullish" if probability > 0.5 else "bearish"
                confidence = abs(probability - 0.5) * 2.0
                signal_strength = (probability - 0.5) * 2.0
                predicted_price = None
                predicted_relative_change = None
                task = "classification"

            result = {
                "instrument": instrument,
                "timestamp": timestamp,
                "task": task,
                "prediction": prediction_label,
                "probability": probability,
                "confidence": confidence,
                "signal_strength": signal_strength,
                "features_used": len(self.feature_names or []),
                "model_version": self.model_type,
                "predicted_relative_change": predicted_relative_change,
                "predicted_price": predicted_price,
            }

            # Log prediction for monitoring
            self._log_prediction(result, features)

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._mock_prediction(instrument, timestamp)

    def _get_full_feature_set(self, instrument: str, feast_features: Optional[Dict]) -> Dict:
        """Get all 70 features needed for the model.

        For now, this uses realistic defaults for market features and integrates
        news sentiment from Feast if available.

        In production, this would fetch real OANDA data and calculate all technical indicators.

        Args:
            instrument: Trading instrument
            feast_features: Features from Feast (mainly news sentiment)

        Returns:
            Dictionary with all 70 features
        """
        # Start with default values for all features
        # These are reasonable mid-range values for technical indicators
        features = {
            # OHLCV
            'open': 6500.0, 'high': 6510.0, 'low': 6490.0, 'close': 6500.0, 'volume': 50000.0,
            # Returns
            'return_1': 0.0001, 'return_5': 0.0005, 'return_10': 0.001,
            # RSI
            'rsi_14': 50.0, 'rsi_20': 50.0,
            # MACD
            'macd': 0.5, 'macd_signal': 0.4, 'macd_histogram': 0.1,
            # Bollinger Bands
            'bb_upper': 6520.0, 'bb_middle': 6500.0, 'bb_lower': 6480.0,
            'bb_width': 40.0, 'bb_position': 0.5,
            # Moving Averages
            'sma_7': 6500.0, 'sma_14': 6500.0, 'sma_21': 6500.0, 'sma_50': 6500.0,
            'ema_7': 6500.0, 'ema_14': 6500.0, 'ema_21': 6500.0,
            # ATR and ADX
            'atr_14': 15.0, 'adx_14': 25.0,
            # Momentum
            'momentum_5': 5.0, 'momentum_10': 10.0, 'momentum_20': 20.0,
            # Rate of Change
            'roc_5': 0.1, 'roc_10': 0.2,
            # Volatility
            'volatility_20': 0.01, 'volatility_50': 0.012,
            # Range indicators
            'hl_range': 20.0, 'hl_range_pct': 0.003, 'hl_range_ma20': 18.0,
            # Spread
            'spread_proxy': 20.0, 'spread_pct': 0.003,
            # Volume features
            'volume_ma20': 50000.0, 'volume_ma50': 50000.0,
            'volume_ratio': 1.0, 'volume_zscore': 0.0,
            # Microstructure
            'price_impact': 0.00001, 'price_impact_ma20': 0.00001,
            'order_flow_imbalance': 0.5, 'illiquidity': 0.2, 'illiquidity_ma20': 0.2,
            # VWAP
            'vwap': 6500.0, 'close_vwap_ratio': 1.0,
            # Volume dynamics
            'volume_velocity': 0.0, 'volume_acceleration': 0.0,
            # Historical volatility
            'hist_vol_20': 0.15, 'hist_vol_50': 0.15,
            # Advanced volatility
            'gk_vol': 0.15, 'parkinson_vol': 0.14, 'rs_vol': 0.15, 'yz_vol': 0.15,
            # Volatility regime
            'vol_of_vol': 0.02, 'vol_regime_low': 0, 'vol_regime_high': 0,
            # Realized range
            'realized_range': 0.003, 'realized_range_ma': 0.003, 'ewma_vol': 0.01,
            # News features (will be updated from Feast or simulated news)
            'news_avg_sentiment': 0.0,
            'news_signal_strength': 0.0,
            'news_article_count': 0,
            'news_quality_score': 0.0,
            'news_age_minutes': 60.0,
            'news_available': 0
        }

        # Update news features from Feast if available
        if feast_features:
            for key, value in feast_features.items():
                if key in features:
                    if isinstance(value, list) and len(value) > 0:
                        features[key] = float(value[0]) if value[0] is not None else features[key]
                    elif value is not None:
                        features[key] = float(value)

        # Also get simulated news sentiment
        simulated_sentiment = self._get_simulated_news_sentiment()
        if simulated_sentiment is not None:
            features['news_avg_sentiment'] = simulated_sentiment
            features['news_signal_strength'] = abs(simulated_sentiment)
            features['news_article_count'] = 1
            features['news_quality_score'] = 0.8
            features['news_age_minutes'] = 5.0
            features['news_available'] = 1

        return features

    def _prepare_features(self, features: Dict) -> List[float]:
        """Prepare feature vector from Feast features.

        Args:
            features: Raw features from Feast

        Returns:
            Feature vector matching model expectations
        """
        feature_vector = []

        # Extract features in the order expected by the model
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0.0)
            if isinstance(value, list) and len(value) > 0:
                value = value[0]
            feature_vector.append(float(value) if value is not None else 0.0)

        return feature_vector

    def _mock_prediction(self, instrument: str, timestamp: datetime) -> Dict:
        """Generate mock prediction when model unavailable.

        Args:
            instrument: Trading instrument
            timestamp: Prediction timestamp

        Returns:
            Mock prediction dictionary
        """
        # Try to read latest simulated news sentiment
        simulated_news_dir = Path("data_clean/bronze/news/simulated")
        avg_sentiment = 0.0
        news_count = 0

        if simulated_news_dir.exists():
            import json
            # Get the 5 most recent news files
            news_files = sorted(simulated_news_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]

            for news_file in news_files:
                try:
                    with open(news_file, 'r') as f:
                        article = json.load(f)
                        sentiment = article.get('sentiment_score', 0.0)
                        avg_sentiment += sentiment
                        news_count += 1
                except:
                    pass

        # Calculate prediction based on news sentiment
        if news_count > 0:
            avg_sentiment = avg_sentiment / news_count
            # Map sentiment [-1, 1] to probability [0.2, 0.8]
            probability = 0.5 + (avg_sentiment * 0.3)  # Scale sentiment to probability
            probability = max(0.2, min(0.8, probability))  # Clamp between 0.2 and 0.8
        else:
            # Fallback to time-based mock if no news
            hour = timestamp.hour
            minute = timestamp.minute
            seed = hour * 60 + minute
            np.random.seed(seed)
            probability = np.random.uniform(0.3, 0.8)

        prediction = "bullish" if probability > 0.5 else "bearish"
        confidence = abs(probability - 0.5) * 2.0
        signal_strength = (probability - 0.5) * 2.0

        return {
            "instrument": instrument,
            "timestamp": timestamp,
            "task": "classification",
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "signal_strength": signal_strength,
            "features_used": len(self.feature_names or []),
            "model_version": f"mock_with_news (n={news_count})",
            "predicted_relative_change": None,
            "predicted_price": None,
        }

    async def get_latest_prediction(self) -> Dict:
        """Get the most recent prediction (for WebSocket streaming).

        Returns:
            Latest prediction dictionary
        """
        return self.predict("SPX500_USD")

    async def get_recent_news(self, limit: int = 10) -> List[Dict]:
        """Get recent news articles with sentiment.

        Args:
            limit: Maximum number of articles

        Returns:
            List of news articles
        """
        # Read from news Gold layer
        news_path = Path("data/news/gold/news_signals/sp500_trading_signals.csv")

        if not news_path.exists():
            return []

        try:
            df = pd.read_csv(news_path)
            df['signal_time'] = pd.to_datetime(df['signal_time'])
            df = df.sort_values('signal_time', ascending=False).head(limit)

            articles = []
            for _, row in df.iterrows():
                articles.append({
                    "time": row['signal_time'].isoformat(),
                    "headline": row.get('latest_headline', 'No headline'),
                    "source": row.get('latest_source', 'Unknown'),
                    "sentiment": float(row.get('avg_sentiment', 0.0)),
                    "impact": self._calculate_impact(row.get('signal_strength', 0)),
                })

            return articles

        except Exception as e:
            logger.error(f"Error reading news: {e}")
            return []

    def _calculate_impact(self, signal_strength: float) -> str:
        """Calculate news impact level.

        Args:
            signal_strength: Signal strength value

        Returns:
            Impact level: "low", "medium", or "high"
        """
        abs_strength = abs(signal_strength)

        if abs_strength > 0.7:
            return "high"
        elif abs_strength > 0.4:
            return "medium"
        else:
            return "low"

    def _infer_task_from_model(self) -> str:
        """Infer prediction task from model capabilities."""
        if self.model is None:
            return "classification"

        if hasattr(self.model, "predict_proba"):
            return "classification"

        return "regression"

    def _relative_change_to_label(self, relative_change: float) -> str:
        """Convert relative price change to directional label."""
        if relative_change > 0:
            return "bullish"
        if relative_change < 0:
            return "bearish"
        return "neutral"

    def _calculate_predicted_price(self, features: Dict, relative_change: float) -> Optional[float]:
        """Estimate predicted price using current price feature if available."""
        current_price = self._get_reference_price(features)
        if current_price is None:
            return None
        return current_price * (1 + relative_change)

    def _get_reference_price(self, features: Dict) -> Optional[float]:
        """Extract current price from available features."""
        candidate_keys: List[str] = []

        if self.price_feature_key:
            candidate_keys.append(self.price_feature_key)

        # Look for price-related keys in the provided features
        candidate_keys.extend([
            key for key in features.keys()
            if any(token in key.lower() for token in ["price", "close", "last", "spot"])
        ])

        seen = set()
        for key in candidate_keys:
            if key in seen:
                continue
            seen.add(key)
            value = features.get(key)
            if isinstance(value, list) and value:
                value = value[0]
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

        return None

    def _get_simulated_news_sentiment(self) -> Optional[float]:
        """Get weighted average sentiment from simulated news files.

        Uses exponential decay weighting where recent news has more impact.
        Weight formula: w_i = exp(-alpha * i) where i is the index (0 = most recent)
        Alpha = 0.5 gives: [1.0, 0.61, 0.37, 0.22, 0.14] for 5 articles
        """
        simulated_news_dir = Path("data_clean/bronze/news/simulated")
        if not simulated_news_dir.exists():
            return None

        try:
            import json
            import math

            # Get most recent 5 news files
            news_files = sorted(simulated_news_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]

            if not news_files:
                return None

            # Exponential decay parameter (higher = more weight on recent news)
            alpha = 0.5

            weighted_sentiment = 0.0
            total_weight = 0.0

            for i, news_file in enumerate(news_files):
                try:
                    with open(news_file, 'r') as f:
                        article = json.load(f)
                        sentiment = article.get('sentiment_score', 0.0)

                        # Calculate exponential decay weight
                        weight = math.exp(-alpha * i)

                        weighted_sentiment += sentiment * weight
                        total_weight += weight
                except:
                    pass

            if total_weight > 0:
                return weighted_sentiment / total_weight
        except:
            pass

        return None

    def _log_prediction(self, prediction: Dict, features: Dict) -> None:
        """Log prediction to disk for monitoring and drift detection.

        Args:
            prediction: Prediction result dictionary
            features: Features used for prediction
        """
        try:
            # Ensure predictions directory exists
            PREDICTIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)

            # Create log entry
            log_entry = {
                "timestamp": prediction["timestamp"].isoformat() if isinstance(prediction["timestamp"], datetime) else prediction["timestamp"],
                "instrument": prediction["instrument"],
                "prediction": prediction["prediction"],
                "probability": prediction.get("probability"),
                "confidence": prediction["confidence"],
                "signal_strength": prediction["signal_strength"],
                "predicted_price": prediction.get("predicted_price"),
                "predicted_relative_change": prediction.get("predicted_relative_change"),
                "task": prediction["task"],
                "model_version": prediction["model_version"],
                "features_used": prediction["features_used"],
                # Add key features for monitoring
                "close": features.get("close"),
                "rsi_14": features.get("rsi_14"),
                "avg_sentiment": features.get("avg_sentiment"),
            }

            # Append to JSONL file
            with open(PREDICTIONS_LOG_FILE, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            logger.debug(f"Logged prediction for {prediction['instrument']} at {prediction['timestamp']}")

        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
