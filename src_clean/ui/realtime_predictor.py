"""
Real-time Event-Driven Prediction Service

Monitors news directory for new files and automatically generates predictions
using real market data + news sentiment features.

Repository Location: fx-ml-pipeline/src_clean/ui/realtime_predictor.py
"""

import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
import numpy as np
from joblib import load
from dotenv import load_dotenv

# OANDA API imports
try:
    import oandapyV20
    from oandapyV20.endpoints.instruments import InstrumentsCandles
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureCalculator:
    """Calculate all 70 features matching training pipeline."""

    def __init__(self):
        self.required_features = [
            "open", "high", "low", "close", "volume",
            "return_1", "return_5", "return_10",
            "rsi_14", "rsi_20",
            "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
            "sma_7", "sma_14", "sma_21", "sma_50",
            "ema_7", "ema_14", "ema_21",
            "atr_14", "adx_14",
            "momentum_5", "momentum_10", "momentum_20",
            "roc_5", "roc_10",
            "volatility_20", "volatility_50",
            "hl_range", "hl_range_pct", "hl_range_ma20",
            "spread_proxy", "spread_pct",
            "volume_ma20", "volume_ma50", "volume_ratio", "volume_zscore",
            "price_impact", "price_impact_ma20",
            "order_flow_imbalance", "illiquidity", "illiquidity_ma20",
            "vwap", "close_vwap_ratio",
            "volume_velocity", "volume_acceleration",
            "hist_vol_20", "hist_vol_50",
            "gk_vol", "parkinson_vol", "rs_vol", "yz_vol",
            "vol_of_vol", "vol_regime_low", "vol_regime_high",
            "realized_range", "realized_range_ma", "ewma_vol",
            "news_avg_sentiment", "news_signal_strength",
            "news_article_count", "news_quality_score",
            "news_age_minutes", "news_available"
        ]

    def calculate_market_features(self, df: pd.DataFrame) -> pd.Series:
        """Calculate all market features from OHLCV data."""
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for feature calculation")
            return None

        latest = df.iloc[-1].copy()

        # Basic OHLCV
        features = {
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'close': latest['close'],
            'volume': latest['volume']
        }

        # Returns
        features['return_1'] = df['close'].pct_change(1).iloc[-1]
        features['return_5'] = df['close'].pct_change(5).iloc[-1]
        features['return_10'] = df['close'].pct_change(10).iloc[-1]

        # RSI
        features['rsi_14'] = self._compute_rsi(df['close'], 14).iloc[-1]
        features['rsi_20'] = self._compute_rsi(df['close'], 20).iloc[-1]

        # MACD
        macd_vals = self._compute_macd(df['close'])
        features['macd'] = macd_vals['macd'].iloc[-1]
        features['macd_signal'] = macd_vals['signal'].iloc[-1]
        features['macd_histogram'] = macd_vals['histogram'].iloc[-1]

        # Bollinger Bands
        bb = self._compute_bollinger_bands(df['close'])
        features['bb_upper'] = bb['upper'].iloc[-1]
        features['bb_middle'] = bb['middle'].iloc[-1]
        features['bb_lower'] = bb['lower'].iloc[-1]
        features['bb_width'] = bb['width'].iloc[-1]
        features['bb_position'] = bb['position'].iloc[-1]

        # Moving Averages
        features['sma_7'] = df['close'].rolling(7).mean().iloc[-1]
        features['sma_14'] = df['close'].rolling(14).mean().iloc[-1]
        features['sma_21'] = df['close'].rolling(21).mean().iloc[-1]
        features['sma_50'] = df['close'].rolling(50).mean().iloc[-1]
        features['ema_7'] = df['close'].ewm(span=7).mean().iloc[-1]
        features['ema_14'] = df['close'].ewm(span=14).mean().iloc[-1]
        features['ema_21'] = df['close'].ewm(span=21).mean().iloc[-1]

        # ATR and ADX
        features['atr_14'] = self._compute_atr(df, 14).iloc[-1]
        features['adx_14'] = self._compute_adx(df, 14).iloc[-1]

        # Momentum
        features['momentum_5'] = df['close'].iloc[-1] - df['close'].iloc[-6]
        features['momentum_10'] = df['close'].iloc[-1] - df['close'].iloc[-11]
        features['momentum_20'] = df['close'].iloc[-1] - df['close'].iloc[-21]

        # Rate of Change
        features['roc_5'] = ((df['close'].iloc[-1] / df['close'].iloc[-6]) - 1) * 100
        features['roc_10'] = ((df['close'].iloc[-1] / df['close'].iloc[-11]) - 1) * 100

        # Volatility
        features['volatility_20'] = df['close'].pct_change().rolling(20).std().iloc[-1]
        features['volatility_50'] = df['close'].pct_change().rolling(50).std().iloc[-1]

        # Range indicators
        features['hl_range'] = latest['high'] - latest['low']
        features['hl_range_pct'] = (latest['high'] - latest['low']) / latest['close']
        features['hl_range_ma20'] = ((df['high'] - df['low'])).rolling(20).mean().iloc[-1]

        # Spread proxies
        features['spread_proxy'] = latest['high'] - latest['low']
        features['spread_pct'] = features['spread_proxy'] / latest['close']

        # Volume features
        features['volume_ma20'] = df['volume'].rolling(20).mean().iloc[-1]
        features['volume_ma50'] = df['volume'].rolling(50).mean().iloc[-1]
        features['volume_ratio'] = latest['volume'] / (features['volume_ma20'] + 1e-10)
        vol_std = df['volume'].rolling(20).std().iloc[-1]
        features['volume_zscore'] = (latest['volume'] - features['volume_ma20']) / (vol_std + 1e-10)

        # Microstructure features
        features['price_impact'] = abs(df['close'].pct_change().iloc[-1]) / (latest['volume'] + 1e-10)
        features['price_impact_ma20'] = (abs(df['close'].pct_change()) / (df['volume'] + 1e-10)).rolling(20).mean().iloc[-1]
        features['order_flow_imbalance'] = (latest['close'] - latest['open']) / (latest['high'] - latest['low'] + 1e-10)
        features['illiquidity'] = abs(df['close'].pct_change().iloc[-1]) / (latest['volume'] + 1e-10) * 1e6
        features['illiquidity_ma20'] = (abs(df['close'].pct_change()) / (df['volume'] + 1e-10) * 1e6).rolling(20).mean().iloc[-1]

        # VWAP
        features['vwap'] = ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).sum() / df['volume'].sum()
        features['close_vwap_ratio'] = latest['close'] / features['vwap']

        # Volume dynamics
        vol_diff = df['volume'].diff()
        features['volume_velocity'] = vol_diff.rolling(10).mean().iloc[-1]
        features['volume_acceleration'] = vol_diff.diff().rolling(10).mean().iloc[-1]

        # Historical volatility
        features['hist_vol_20'] = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        features['hist_vol_50'] = df['close'].pct_change().rolling(50).std().iloc[-1] * np.sqrt(252)

        # Advanced volatility estimators
        features['gk_vol'] = self._garman_klass_vol(df).iloc[-1]
        features['parkinson_vol'] = self._parkinson_vol(df).iloc[-1]
        features['rs_vol'] = self._rogers_satchell_vol(df).iloc[-1]
        features['yz_vol'] = self._yang_zhang_vol(df).iloc[-1]

        # Volatility regime
        vol = df['close'].pct_change().rolling(20).std()
        features['vol_of_vol'] = vol.rolling(20).std().iloc[-1]
        vol_20 = vol.iloc[-1]
        vol_50 = df['close'].pct_change().rolling(50).std().iloc[-1]
        features['vol_regime_low'] = 1 if vol_20 < vol_50 * 0.8 else 0
        features['vol_regime_high'] = 1 if vol_20 > vol_50 * 1.2 else 0

        # Realized range
        rr = (df['high'] - df['low']) / df['close']
        features['realized_range'] = rr.iloc[-1]
        features['realized_range_ma'] = rr.rolling(20).mean().iloc[-1]

        # EWMA volatility
        features['ewma_vol'] = df['close'].pct_change().ewm(span=20).std().iloc[-1]

        return pd.Series(features)

    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, prices: pd.Series) -> Dict:
        """Compute MACD."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return {'macd': macd, 'signal': signal, 'histogram': histogram}

    def _compute_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        """Compute Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        width = upper - lower
        position = (prices - lower) / (width + 1e-10)
        return {'upper': upper, 'middle': middle, 'lower': lower, 'width': width, 'position': position}

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute ATR."""
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['close'].shift())
        lc = abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _compute_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute ADX (simplified)."""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        atr = self._compute_atr(df, period)
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        return dx.rolling(period).mean()

    def _garman_klass_vol(self, df: pd.DataFrame) -> pd.Series:
        """Garman-Klass volatility."""
        hl = np.log(df['high'] / df['low'])
        co = np.log(df['close'] / df['open'])
        return np.sqrt((0.5 * hl**2 - (2*np.log(2)-1) * co**2).rolling(20).mean()) * np.sqrt(252)

    def _parkinson_vol(self, df: pd.DataFrame) -> pd.Series:
        """Parkinson volatility."""
        hl = np.log(df['high'] / df['low'])
        return np.sqrt((1/(4*np.log(2))) * (hl**2).rolling(20).mean()) * np.sqrt(252)

    def _rogers_satchell_vol(self, df: pd.DataFrame) -> pd.Series:
        """Rogers-Satchell volatility."""
        ho = np.log(df['high'] / df['open'])
        hc = np.log(df['high'] / df['close'])
        lo = np.log(df['low'] / df['open'])
        lc = np.log(df['low'] / df['close'])
        return np.sqrt((ho * hc + lo * lc).rolling(20).mean()) * np.sqrt(252)

    def _yang_zhang_vol(self, df: pd.DataFrame) -> pd.Series:
        """Yang-Zhang volatility (simplified)."""
        co = np.log(df['close'] / df['open'])
        oc = np.log(df['open'] / df['close'].shift())
        rs_vol = self._rogers_satchell_vol(df) / np.sqrt(252)
        close_vol = co.rolling(20).std()
        open_vol = oc.rolling(20).std()
        k = 0.34 / (1.34 + (20 + 1) / (20 - 1))
        return np.sqrt(open_vol**2 + k * close_vol**2 + (1-k) * rs_vol**2) * np.sqrt(252)

    def load_latest_news(self, news_dir: Path, lookback_hours: int = 6) -> Dict:
        """Load latest news sentiment features."""
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

        news_files = sorted(news_dir.glob("simulated_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not news_files:
            return {
                'news_avg_sentiment': 0.0,
                'news_signal_strength': 0.0,
                'news_article_count': 0,
                'news_quality_score': 0.0,
                'news_age_minutes': np.nan,
                'news_available': 0
            }

        # Get most recent news
        recent_news = []
        for news_file in news_files:
            file_time = datetime.fromtimestamp(news_file.stat().st_mtime)
            if file_time < cutoff_time:
                break
            try:
                with open(news_file) as f:
                    article = json.load(f)
                    recent_news.append(article)
            except Exception as e:
                logger.warning(f"Error loading {news_file}: {e}")

        if not recent_news:
            return {
                'news_avg_sentiment': 0.0,
                'news_signal_strength': 0.0,
                'news_article_count': 0,
                'news_quality_score': 0.0,
                'news_age_minutes': np.nan,
                'news_available': 0
            }

        # Calculate aggregated news features
        sentiments = [a.get('sentiment_score', 0.0) for a in recent_news]
        latest_article = recent_news[0]
        latest_time = datetime.fromisoformat(latest_article.get('streamed_at', datetime.now().isoformat()))
        age_minutes = (datetime.now() - latest_time).total_seconds() / 60

        return {
            'news_avg_sentiment': np.mean(sentiments),
            'news_signal_strength': abs(np.mean(sentiments)),
            'news_article_count': len(recent_news),
            'news_quality_score': 1.0 if not latest_article.get('mock', False) else 0.5,
            'news_age_minutes': age_minutes,
            'news_available': 1
        }

    def calculate_all_features(self, market_df: pd.DataFrame, news_dir: Path) -> Optional[pd.Series]:
        """Calculate all 70 features."""
        try:
            # Market features (64 features)
            market_features = self.calculate_market_features(market_df)
            if market_features is None:
                return None

            # News features (6 features)
            news_features = self.load_latest_news(news_dir)

            # Combine
            all_features = pd.concat([market_features, pd.Series(news_features)])

            # Ensure correct order
            feature_vector = pd.Series(dtype=float)
            for feat in self.required_features:
                if feat in all_features:
                    feature_vector[feat] = all_features[feat]
                else:
                    feature_vector[feat] = 0.0
                    logger.warning(f"Missing feature: {feat}, using 0.0")

            return feature_vector

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None


class NewsEventHandler(FileSystemEventHandler):
    """Handles new news file events."""

    def __init__(self, predictor):
        self.predictor = predictor

    def on_created(self, event):
        """Trigger prediction when new news file is created."""
        if event.is_directory:
            return
        if event.src_path.endswith('.json'):
            logger.info(f"New news detected: {event.src_path}")
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            self.predictor.trigger_prediction(news_file_path=event.src_path)


class RealtimePredictor:
    """Event-driven prediction service."""

    def __init__(self,
                 model_dir: Path = Path("data_clean/models"),
                 news_dir: Path = Path("data_clean/bronze/news/simulated"),
                 predictions_file: Path = Path("data_clean/predictions/latest_prediction.json"),
                 history_file: Path = Path("data_clean/predictions/prediction_history.json")):
        self.model_dir = model_dir
        self.news_dir = news_dir
        self.predictions_file = predictions_file
        self.history_file = history_file
        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_names = None
        self.feature_calculator = FeatureCalculator()

        self.load_model()

    def load_model(self):
        """Load latest trained model."""
        try:
            model_files = list(self.model_dir.glob("xgboost_classification_*.pkl"))
            if not model_files:
                logger.error("No trained models found")
                return

            latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            self.model = load(latest_model_path)

            # Load feature names
            features_file = latest_model_path.with_name(f"{latest_model_path.stem}_features.json")
            with open(features_file) as f:
                self.feature_names = json.load(f)['features']

            logger.info(f"Loaded model: {latest_model_path.name}")
            logger.info(f"Features: {len(self.feature_names)}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def get_oanda_market_data(self, instrument: str = "SPX500_USD", granularity: str = "M1", count: int = 200) -> pd.DataFrame:
        """Fetch real market data from OANDA API."""
        if not OANDA_AVAILABLE:
            logger.warning("OANDA library not available, using mock data")
            return self.get_mock_market_data()

        try:
            # Initialize OANDA API
            token = os.getenv("OANDA_TOKEN")
            env = os.getenv("OANDA_ENV", "practice")

            if not token:
                logger.warning("OANDA_TOKEN not found, using mock data")
                return self.get_mock_market_data()

            api = oandapyV20.API(access_token=token, environment=env)

            # Fetch candles
            params = {
                "granularity": granularity,
                "count": count,
                "price": "M"  # Mid prices
            }
            request = InstrumentsCandles(instrument=instrument, params=params)
            response = api.request(request)

            candles = response.get("candles", [])
            if not candles:
                logger.warning("No candles received from OANDA, using mock data")
                return self.get_mock_market_data()

            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append({
                    'time': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle.get('volume', 0))
                })

            df = pd.DataFrame(data)
            logger.info(f"Fetched {len(df)} candles from OANDA for {instrument}")
            return df

        except Exception as e:
            logger.error(f"Error fetching OANDA data: {e}, using mock data")
            return self.get_mock_market_data()

    def get_mock_market_data(self) -> pd.DataFrame:
        """Generate mock market data for testing (fallback)."""
        dates = pd.date_range(end=datetime.now(), periods=200, freq='1min')
        base_price = 4500
        prices = base_price + np.cumsum(np.random.randn(200) * 2)

        return pd.DataFrame({
            'time': dates,
            'open': prices + np.random.randn(200) * 0.5,
            'high': prices + abs(np.random.randn(200) * 2),
            'low': prices - abs(np.random.randn(200) * 2),
            'close': prices,
            'volume': np.random.randint(10000, 100000, 200)
        })

    def trigger_prediction(self, news_file_path: Optional[str] = None):
        """Generate prediction with real features from OANDA."""
        if self.model is None:
            logger.error("Model not loaded")
            return

        try:
            # Load news article details if provided
            news_article = None
            if news_file_path:
                try:
                    with open(news_file_path, 'r') as f:
                        news_data = json.load(f)
                        news_article = {
                            'headline': news_data.get('headline', 'N/A'),
                            'source': news_data.get('source', 'N/A'),
                            'published_at': news_data.get('published_at', 'N/A'),
                            'sentiment_score': news_data.get('sentiment_score', 0),
                            'sentiment_type': news_data.get('sentiment_type', 'neutral'),
                            'content': news_data.get('content', 'N/A')
                        }
                except Exception as e:
                    logger.warning(f"Could not load news article: {e}")

            # Get real market data from OANDA (24/5 S&P 500 futures)
            market_df = self.get_oanda_market_data()

            # Calculate all features
            features = self.feature_calculator.calculate_all_features(market_df, self.news_dir)
            if features is None:
                logger.error("Failed to calculate features")
                return

            # Make prediction
            feature_vector = features.values.reshape(1, -1)
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]

            # Prepare result
            result = {
                'timestamp': datetime.now().isoformat(),
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'UP': float(probabilities[1]),
                    'DOWN': float(probabilities[0])
                },
                'trigger': 'new_news_article',
                'signal_strength': 'Strong' if max(probabilities) > 0.7 else ('Moderate' if max(probabilities) > 0.6 else 'Weak'),
                'features_calculated': len(features),
                'news_available': int(features['news_available']),
                'news_sentiment': float(features['news_avg_sentiment'])
            }

            # Add news article details if available
            if news_article:
                result['news_article'] = news_article

            # Save prediction
            with open(self.predictions_file, 'w') as f:
                json.dump(result, f, indent=2)

            # Save to history
            self._save_to_history(result)

            logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)

    def _save_to_history(self, prediction: Dict):
        """Save prediction to history file."""
        try:
            # Load existing history
            history = []
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)

            # Add new prediction (keep only essential fields)
            history_entry = {
                'timestamp': prediction['timestamp'],
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'prob_up': prediction['probabilities']['UP'],
                'prob_down': prediction['probabilities']['DOWN'],
                'news_sentiment': prediction.get('news_sentiment', 0)
            }

            history.append(history_entry)

            # Keep only last 100 predictions
            if len(history) > 100:
                history = history[-100:]

            # Save updated history
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)

            logger.debug(f"Saved to history. Total predictions: {len(history)}")

        except Exception as e:
            logger.warning(f"Error saving to history: {e}")

    def start_watching(self):
        """Start watching news directory for events."""
        logger.info(f"Watching for news in: {self.news_dir}")

        event_handler = NewsEventHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.news_dir), recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


if __name__ == "__main__":
    predictor = RealtimePredictor()
    predictor.start_watching()
