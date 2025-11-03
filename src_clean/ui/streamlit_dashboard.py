"""
Enhanced Streamlit Dashboard for S&P500 ML Prediction Pipeline

Features:
- Real-time OANDA price data with forecast projections
- ML predictions with confidence scores
- Feature importance visualization
- Model metrics dashboard
- News sentiment integration with snippets
- Historical prediction performance tracking
- News sentiment timeline
- Price forecast with confidence bands
"""

import streamlit as st
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from joblib import load
import glob

# Try to import OANDA (optional - fallback to mock data)
try:
    import oandapyV20
    import oandapyV20.endpoints.pricing as pricing
    import oandapyV20.endpoints.instruments as instruments
    from dotenv import load_dotenv
    load_dotenv()
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False

# Try to import TextBlob for sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Configuration
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_TOKEN = os.getenv("OANDA_TOKEN")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")
MODEL_DIR = Path("data_clean/models")
GOLD_DIR = Path("data_clean/gold")
PREDICTIONS_FILE = Path("data_clean/predictions/latest_prediction.json")
PREDICTIONS_HISTORY_FILE = Path("data_clean/predictions/prediction_history.json")
NEWS_DIR = Path("data_clean/bronze/news/simulated")

# Initialize OANDA API if available
if OANDA_AVAILABLE and OANDA_TOKEN:
    OANDA_API = oandapyV20.API(access_token=OANDA_TOKEN, environment=OANDA_ENV)
else:
    OANDA_API = None
    if not OANDA_AVAILABLE:
        st.sidebar.warning("⚠️ OANDA not available - using mock data for charts")
    elif not OANDA_TOKEN:
        st.sidebar.warning("⚠️ OANDA_TOKEN not configured - using mock data for charts")


@st.cache_resource
def load_latest_model():
    """Load the most recent trained model (regression or classification)."""
    try:
        # Find latest model - prefer regression models for price prediction
        regression_files = list(MODEL_DIR.glob("xgboost_regression_*.pkl"))
        classification_files = list(MODEL_DIR.glob("xgboost_classification_*.pkl"))

        model_files = regression_files + classification_files
        if not model_files:
            return None, None, None

        latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)

        # Load model
        model_bundle = load(latest_model_path)
        if isinstance(model_bundle, dict):
            model = model_bundle.get('model')
            feature_names = model_bundle.get('feature_names', [])
        else:
            model = model_bundle
            feature_names = getattr(model, 'feature_names_in_', [])

        # Try to load metrics if available
        model_name = latest_model_path.stem
        metrics_path = MODEL_DIR / f"{model_name}_metrics.json"

        metrics = None
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            # Create placeholder metrics for regression model
            metrics = {
                'model_type': 'regression' if 'regression' in model_name else 'classification',
                'model_name': model_name,
                'n_features': len(feature_names),
                'trained_at': latest_model_path.stat().st_mtime
            }

        return model, feature_names, metrics
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def get_latest_price(account_id, instrument="SPX500_USD"):
    """Get latest price from OANDA."""
    if not OANDA_API or not account_id:
        # Mock data
        return 4521.50 + np.random.randn() * 10

    try:
        params = {"instruments": instrument}
        request = pricing.PricingInfo(accountID=account_id, params=params)
        response = OANDA_API.request(request)
        prices = response['prices'][0]
        bid = float(prices['bids'][0]['price'])
        ask = float(prices['asks'][0]['price'])
        return (bid + ask) / 2
    except Exception as e:
        st.error(f"Error fetching price: {e}")
        return None


def get_candles(instrument="SPX500_USD", granularity="H1", count=100):
    """Get historical candles from OANDA."""
    if not OANDA_API:
        # Generate mock candles with realistic S&P 500 prices
        dates = pd.date_range(end=datetime.now(), periods=count, freq='h')
        base_price = 6500  # More realistic current S&P 500 level
        prices = base_price + np.cumsum(np.random.randn(count) * 10)

        return pd.DataFrame({
            'time': dates,
            'open': prices + np.random.randn(count) * 2,
            'high': prices + abs(np.random.randn(count) * 5),
            'low': prices - abs(np.random.randn(count) * 5),
            'close': prices,
            'volume': np.random.randint(10000, 100000, count)
        })

    try:
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M"
        }
        request = instruments.InstrumentsCandles(instrument=instrument, params=params)
        response = OANDA_API.request(request)
        time.sleep(0.5)
        candles = response.get("candles", [])

        if not candles:
            # Fallback to mock data
            st.warning("No candles received from OANDA, using mock data")
            return get_candles(instrument, granularity, count)

        df = pd.DataFrame([
            {
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": int(c.get("volume", 0))
            } for c in candles
        ])
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        st.error(f"Error fetching candles: {e}")
        # Fallback to mock data on error
        dates = pd.date_range(end=datetime.now(), periods=count, freq='H')
        base_price = 4500
        prices = base_price + np.cumsum(np.random.randn(count) * 10)
        return pd.DataFrame({
            'time': dates,
            'open': prices + np.random.randn(count) * 2,
            'high': prices + abs(np.random.randn(count) * 5),
            'low': prices - abs(np.random.randn(count) * 5),
            'close': prices,
            'volume': np.random.randint(10000, 100000, count)
        })


def calculate_technical_features(df):
    """Calculate basic technical features for prediction."""
    # Simple features for demo
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    return df


def load_latest_prediction():
    """Load the latest prediction from FastAPI."""
    try:
        import requests
        # Try Docker network first, then localhost for local development
        fastapi_url = os.getenv('FASTAPI_URL', 'http://fastapi:8000')
        # Fallback to localhost if FASTAPI_URL not set
        if fastapi_url == 'http://fastapi:8000':
            try:
                response = requests.post(f'{fastapi_url}/predict',
                                        json={'instrument': 'SPX500_USD'},
                                        timeout=2)
            except:
                # Fallback to localhost
                fastapi_url = 'http://localhost:8000'
                response = requests.post(f'{fastapi_url}/predict',
                                        json={'instrument': 'SPX500_USD'},
                                        timeout=5)
        else:
            response = requests.post(f'{fastapi_url}/predict',
                                    json={'instrument': 'SPX500_USD'},
                                    timeout=5)

        if response.status_code == 200:
            pred_data = response.json()

            # Convert to display format
            task = pred_data.get('task', 'classification')

            if task == 'regression':
                # For regression model
                return {
                    'prediction': pred_data['prediction'],
                    'confidence': pred_data['confidence'],
                    'prob_up': 0.5 + (pred_data['signal_strength'] / 2.0),  # Convert to 0-1 range
                    'prob_down': 0.5 - (pred_data['signal_strength'] / 2.0),
                    'timestamp': pred_data['timestamp'],
                    'trigger': 'api_call',
                    'features_used': pred_data['features_used'],
                    'news_article': None,
                    'news_sentiment': pred_data['signal_strength'],
                    'predicted_price': pred_data.get('predicted_price'),
                    'predicted_change': pred_data.get('predicted_relative_change', 0) * 100  # Convert to %
                }
            else:
                # For classification model
                probability = pred_data.get('probability', 0.5)
                return {
                    'prediction': pred_data['prediction'],
                    'confidence': pred_data['confidence'],
                    'prob_up': probability,
                    'prob_down': 1.0 - probability,
                    'timestamp': pred_data['timestamp'],
                    'trigger': 'api_call',
                    'features_used': pred_data['features_used'],
                    'news_article': None,
                    'news_sentiment': pred_data['signal_strength']
                }
        else:
            return None

    except Exception as e:
        # Fallback: try to load from file
        try:
            if PREDICTIONS_FILE.exists():
                with open(PREDICTIONS_FILE, 'r') as f:
                    pred_data = json.load(f)
                return {
                    'prediction': pred_data['prediction'],
                    'confidence': pred_data['confidence'],
                    'prob_up': pred_data['probabilities']['UP'],
                    'prob_down': pred_data['probabilities']['DOWN'],
                    'timestamp': pred_data['timestamp'],
                    'trigger': pred_data.get('trigger', 'file'),
                    'features_used': pred_data.get('features_calculated', 0),
                    'news_article': pred_data.get('news_article'),
                    'news_sentiment': pred_data.get('news_sentiment', 0)
                }
        except:
            pass
        return None


def analyze_custom_news_sentiment(news_text: str) -> dict:
    """Analyze sentiment of custom news text and return prediction impact."""
    if not news_text or not news_text.strip():
        return None

    if not TEXTBLOB_AVAILABLE:
        st.warning("TextBlob not available. Install with: pip install textblob")
        return None

    try:
        # Analyze sentiment using TextBlob
        blob = TextBlob(news_text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1

        # Map sentiment to trading signal
        if polarity > 0.1:
            sentiment_type = "positive"
            prediction = "UP"
            signal_strength = min(abs(polarity), 1.0)
        elif polarity < -0.1:
            sentiment_type = "negative"
            prediction = "DOWN"
            signal_strength = min(abs(polarity), 1.0)
        else:
            sentiment_type = "neutral"
            prediction = "NEUTRAL"
            signal_strength = 0.1

        # Calculate confidence based on both polarity and subjectivity
        # Higher subjectivity means less confident
        confidence = signal_strength * (1 - subjectivity * 0.5)

        return {
            'news_text': news_text[:200] + '...' if len(news_text) > 200 else news_text,
            'sentiment_type': sentiment_type,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'prediction': prediction,
            'confidence': confidence,
            'signal_strength': signal_strength,
            'prob_up': 0.5 + (polarity * 0.5) if polarity > 0 else 0.5 * (1 + polarity),
            'prob_down': 0.5 - (polarity * 0.5) if polarity > 0 else 0.5 * (1 - polarity),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return None


def load_prediction_history(limit=20):
    """Load prediction history."""
    try:
        if not PREDICTIONS_HISTORY_FILE.exists():
            return []

        with open(PREDICTIONS_HISTORY_FILE, 'r') as f:
            history = json.load(f)

        return history[-limit:] if len(history) > limit else history
    except Exception as e:
        st.warning(f"No prediction history available: {e}")
        return []


def load_recent_news(limit=10):
    """Load recent news articles."""
    try:
        news_files = sorted(NEWS_DIR.glob("simulated_*.json"),
                          key=lambda p: p.stat().st_mtime,
                          reverse=True)[:limit]

        news_list = []
        for news_file in news_files:
            try:
                with open(news_file) as f:
                    article = json.load(f)
                    article['file_time'] = datetime.fromtimestamp(news_file.stat().st_mtime)
                    news_list.append(article)
            except Exception as e:
                continue

        return news_list
    except Exception as e:
        return []


def generate_price_forecast(df, prediction_result, forecast_hours=4):
    """Generate price forecast based on ML prediction and recent trend."""
    try:
        if df is None or df.empty:
            return None

        current_price = df['close'].iloc[-1]
        recent_volatility = df['close'].pct_change().rolling(20).std().iloc[-1]

        # Default to neutral if no prediction
        if not prediction_result:
            expected_move_pct = 0.0
        # Check if we have a predicted price from regression model
        elif 'predicted_price' in prediction_result and prediction_result['predicted_price']:
            # Use actual predicted price from regression model
            target_price = prediction_result['predicted_price']
            expected_move_pct = (target_price - current_price) / current_price
        elif 'predicted_change' in prediction_result and prediction_result['predicted_change']:
            # Use predicted change percentage
            expected_move_pct = prediction_result['predicted_change'] / 100.0
        else:
            # Fallback to classification-based prediction
            pred_direction = prediction_result['prediction'].lower()
            if pred_direction in ['neutral', 'hold']:
                expected_move_pct = 0.0
            else:
                direction = 1 if pred_direction in ['up', 'bullish'] else -1
                confidence = prediction_result.get('confidence', 0.5)
                expected_move_pct = direction * confidence * recent_volatility * np.sqrt(forecast_hours)

        # Generate forecast points
        forecast_periods = 12  # 12 points for smooth line
        forecast_times = pd.date_range(
            start=df['time'].iloc[-1],
            periods=forecast_periods + 1,
            freq='20min'
        )[1:]

        # Generate price path with some randomness
        forecast_prices = []
        upper_band = []
        lower_band = []

        for i in range(1, forecast_periods + 1):
            progress = i / forecast_periods
            # Base forecast with slight drift
            base_move = current_price * (1 + expected_move_pct * progress)
            forecast_prices.append(base_move)

            # Confidence bands (wider as we go further)
            band_width = current_price * recent_volatility * np.sqrt(progress) * 2
            upper_band.append(base_move + band_width)
            lower_band.append(base_move - band_width)

        return {
            'times': forecast_times,
            'prices': forecast_prices,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'current_price': current_price
        }
    except Exception as e:
        st.warning(f"Could not generate forecast: {e}")
        return None


def truncate_text(text, max_length=150):
    """Truncate text to max_length and add ellipsis."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + '...'


def make_prediction(model, features, feature_names, latest_data):
    """
    DEPRECATED: Manual prediction function using random features.
    Use load_latest_prediction() for event-driven predictions instead.
    This function is kept for backward compatibility only.
    """
    try:
        # Prepare feature vector (simplified - would need full feature calculation)
        # For now, use mock features matching training
        feature_vector = np.random.rand(1, len(feature_names))

        # Predict
        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0]

        return {
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': float(max(probability)),
            'prob_up': float(probability[1]),
            'prob_down': float(probability[0])
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def plot_candlestick(df, forecast_data=None):
    """Create candlestick chart with volume and optional forecast."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('S&P 500 Price with Forecast', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add forecast if provided
    if forecast_data:
        # Forecast line (dotted)
        fig.add_trace(
            go.Scatter(
                x=forecast_data['times'],
                y=forecast_data['prices'],
                mode='lines',
                name='Forecast',
                line=dict(color='orange', width=2, dash='dot'),
                showlegend=True
            ),
            row=1, col=1
        )

        # Upper confidence band
        fig.add_trace(
            go.Scatter(
                x=forecast_data['times'],
                y=forecast_data['upper_band'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Lower confidence band (fill between)
        fig.add_trace(
            go.Scatter(
                x=forecast_data['times'],
                y=forecast_data['lower_band'],
                mode='lines',
                name='Confidence Band',
                line=dict(width=0),
                fillcolor='rgba(255, 165, 0, 0.2)',
                fill='tonexty',
                showlegend=True
            ),
            row=1, col=1
        )

    # Volume
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green'
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df['time'], y=df['volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True if forecast_data else False,
        title_text="S&P 500 Market Data with ML Forecast"
    )

    return fig


def plot_feature_importance(model, feature_names):
    """Plot feature importance from model."""
    try:
        # Get feature importances - handle both dict bundles and direct models
        if isinstance(model, dict):
            actual_model = model.get('model', model)
        else:
            actual_model = model

        # Get feature importances
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
        else:
            # If model doesn't have feature_importances_, return None
            return None

        # Ensure we have matching lengths
        if len(feature_names) != len(importances):
            st.warning(f"Feature name count ({len(feature_names)}) doesn't match importance count ({len(importances)})")
            return None

        # Create DataFrame with explicit conversion to list
        importance_df = pd.DataFrame({
            'feature': list(feature_names),
            'importance': importances.tolist() if hasattr(importances, 'tolist') else list(importances)
        })

        # Sort and get top 20
        importance_df = importance_df.sort_values('importance', ascending=False).head(20)

        # Create horizontal bar chart
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 20 Feature Importance',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig
    except Exception as e:
        st.error(f"Error plotting feature importance: {e}")
        return None


def display_model_metrics(metrics):
    """Display model performance metrics (regression or classification)."""
    model_type = metrics.get('model_type', 'unknown')

    if model_type == 'regression':
        # Regression metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            rmse = metrics.get('rmse', metrics.get('test_rmse', 0))
            st.metric("RMSE", f"{rmse:.4f}")
        with col2:
            mae = metrics.get('mae', metrics.get('test_mae', 0))
            st.metric("MAE", f"{mae:.4f}")
        with col3:
            r2 = metrics.get('r2', metrics.get('test_r2', 0))
            st.metric("R² Score", f"{r2:.4f}")
        with col4:
            mape = metrics.get('mape', 0)
            if mape > 0:
                st.metric("MAPE", f"{mape:.2%}")
            else:
                st.metric("Features", metrics.get('n_features', 0))
    else:
        # Classification metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        with col2:
            st.metric("AUC", f"{metrics.get('auc', 0):.4f}")
        with col3:
            st.metric("CV Mean", f"{metrics.get('cv_mean', 0):.4f}")
        with col4:
            st.metric("CV Std", f"{metrics.get('cv_std', 0):.4f}")


def plot_news_sentiment_timeline(news_list):
    """Plot news sentiment over time."""
    if not news_list:
        return None

    df = pd.DataFrame([
        {
            'time': article['file_time'],
            'sentiment': article.get('sentiment_score', 0),
            'headline': truncate_text(article.get('headline', 'N/A'), 50)
        }
        for article in news_list
    ])

    fig = go.Figure()

    # Sentiment line
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['sentiment'],
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        text=df['headline'],
        hovertemplate='<b>%{text}</b><br>Sentiment: %{y:.2f}<br>%{x}<extra></extra>'
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="News Sentiment Timeline",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        height=300,
        hovermode='closest'
    )

    return fig


def plot_prediction_history(history):
    """Plot prediction history and accuracy."""
    if not history:
        return None

    df = pd.DataFrame(history)
    df['time'] = pd.to_datetime(df['timestamp'])
    df['correct'] = df['prediction'] == df.get('actual', 'UNKNOWN')

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Prediction Confidence Over Time', 'Prediction Outcomes'),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15
    )

    # Confidence over time
    colors = ['green' if p == 'UP' else 'red' for p in df['prediction']]
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(width=2),
            marker=dict(size=8, color=colors),
            hovertemplate='<b>%{text}</b><br>Confidence: %{y:.1%}<extra></extra>',
            text=df['prediction']
        ),
        row=1, col=1
    )

    # Prediction outcomes (if actual data available)
    if 'actual' in df.columns:
        outcome_counts = df.groupby(['prediction', 'correct']).size().reset_index(name='count')
        fig.add_trace(
            go.Bar(
                x=outcome_counts['prediction'],
                y=outcome_counts['count'],
                name='Outcomes',
                marker_color=['green' if c else 'red' for c in outcome_counts['correct']]
            ),
            row=2, col=1
        )

    fig.update_layout(height=500, showlegend=False)
    fig.update_yaxes(title_text="Confidence", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig


def display_prediction_gauge(pred_result):
    """Display prediction confidence as a gauge."""
    if not pred_result:
        return

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_result['prob_up'] * 100,
        title={'text': f"Prediction: {pred_result['prediction']}"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkgreen" if pred_result['prediction'] == 'UP' else "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="S&P 500 ML Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚀 S&P 500 ML Prediction Dashboard")
    st.markdown("Real-time price prediction using XGBoost and technical features")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    instrument = st.sidebar.text_input("Instrument", "SPX500_USD")
    granularity = st.sidebar.selectbox("Granularity", ["M15", "H1", "H4", "D"], index=1)
    candle_count = st.sidebar.slider("Candle Count", 50, 500, 100)
    refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 5, 300, 5)

    # Load model
    st.sidebar.header("🤖 Model Status")
    model, feature_names, metrics = load_latest_model()

    if model is None:
        st.sidebar.warning("⚠️ No local model file")
        st.sidebar.info("Using FastAPI model service")
    else:
        model_type = metrics.get('model_type', 'unknown') if metrics else 'unknown'
        model_name = metrics.get('model_name', 'N/A') if metrics else 'N/A'

        st.sidebar.success("✅ Model loaded")
        st.sidebar.text(f"Type: {model_type.title()}")
        st.sidebar.text(f"Features: {len(feature_names)}")
        st.sidebar.caption(f"Model: {model_name[:30]}...")

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Live Trading",
        "🎯 Predictions",
        "📰 News & Sentiment",
        "📊 Model Performance",
        "🔍 Feature Analysis"
    ])

    with tab1:
        st.header(f"Live Market Data: {instrument}")

        # Load prediction for forecast
        pred_result = load_latest_prediction()

        # Current price
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            current_price = get_latest_price(OANDA_ACCOUNT_ID, instrument)
            if current_price:
                st.metric("Current Price", f"${current_price:.2f}", delta=None)

        # Fetch and display candles
        df = get_candles(instrument, granularity, candle_count)
        if df is not None and not df.empty:
            # Generate forecast (even with neutral prediction if no data)
            forecast_hours = st.slider("Forecast Horizon (hours)", 1, 12, 4, key='forecast_hours')

            # Use prediction if available, otherwise create neutral prediction for forecast
            if pred_result:
                forecast_pred = pred_result
                st.info(f"🔮 **ML Forecast**: Price expected to move **{pred_result['prediction']}** "
                       f"with **{pred_result['confidence']:.1%}** confidence over next {forecast_hours} hours")
            else:
                # Create neutral prediction for visualization
                forecast_pred = {
                    'prediction': 'neutral',
                    'confidence': 0.5,
                    'predicted_price': None,
                    'predicted_change': 0.0
                }
                st.info(f"🔮 **Neutral Forecast**: Awaiting prediction data. Showing neutral price projection.")

            forecast_data = generate_price_forecast(df, forecast_pred, forecast_hours)

            # Plot with forecast
            fig = plot_candlestick(df, forecast_data)
            st.plotly_chart(fig, use_container_width=True)

            # Recent data table
            with st.expander("📋 Recent Data"):
                st.dataframe(df.tail(10))

    with tab2:
        st.header("🎯 Event-Driven ML Predictions")

        # Add refresh button and auto-refresh toggle
        col_refresh, col_auto = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.rerun()
        with col_auto:
            auto_refresh_predictions = st.checkbox("🔄 Auto-refresh predictions every 10 seconds", value=False)

        st.markdown("---")

        # Load latest prediction from event-driven system
        pred_result = load_latest_prediction()

        if pred_result:
            # Display prediction timestamp and trigger info
            col_time, col_trigger, col_features = st.columns(3)
            with col_time:
                pred_time = datetime.fromisoformat(pred_result['timestamp'])
                time_ago = datetime.now() - pred_time
                st.metric("Last Prediction", pred_time.strftime("%H:%M:%S"),
                         delta=f"{int(time_ago.total_seconds())}s ago")
            with col_trigger:
                st.metric("Triggered By", pred_result['trigger'].replace('_', ' ').title())
            with col_features:
                st.metric("Features Calculated", pred_result['features_used'])

            st.markdown("---")

            # Display prediction
            col1, col2 = st.columns(2)

            with col1:
                display_prediction_gauge(pred_result)

            with col2:
                st.subheader("Prediction Details")
                st.metric("Direction", pred_result['prediction'],
                         delta="Bullish" if pred_result['prediction'] == 'UP' else "Bearish")
                st.metric("Confidence", f"{pred_result['confidence']:.1%}")
                st.metric("Probability UP", f"{pred_result['prob_up']:.1%}")
                st.metric("Probability DOWN", f"{pred_result['prob_down']:.1%}")

                # Recommendation
                if pred_result['confidence'] > 0.7:
                    st.success(f"🟢 Strong signal: {pred_result['prediction']}")
                elif pred_result['confidence'] > 0.6:
                    st.info(f"🟡 Moderate signal: {pred_result['prediction']}")
                else:
                    st.warning("⚪ Weak signal - trade with caution")

            st.markdown("---")

            # Display news snippet preview with sentiment
            if 'news_article' in pred_result and pred_result['news_article']:
                news = pred_result['news_article']

                # Sentiment badge
                sentiment_type = news.get('sentiment_type', 'neutral')
                sentiment_score = news.get('sentiment_score', 0)

                if sentiment_type == 'positive':
                    sentiment_color = "🟢"
                    sentiment_badge = f"🟢 Positive ({sentiment_score:.2f})"
                elif sentiment_type == 'negative':
                    sentiment_color = "🔴"
                    sentiment_badge = f"🔴 Negative ({sentiment_score:.2f})"
                else:
                    sentiment_color = "⚪"
                    sentiment_badge = f"⚪ Neutral ({sentiment_score:.2f})"

                # News snippet card with preview
                st.markdown("### 📰 News Trigger")

                with st.container():
                    st.markdown(f"**{sentiment_color} {news.get('headline', 'N/A')}**")

                    col_meta1, col_meta2 = st.columns(2)
                    with col_meta1:
                        st.caption(f"Source: {news.get('source', 'N/A').replace('_', ' ').title()}")
                    with col_meta2:
                        st.caption(f"Published: {news.get('published_at', 'N/A')[:10]}")

                    st.markdown(f"**Sentiment:** {sentiment_badge}")

                    # Show snippet preview
                    content = news.get('content', 'N/A')
                    if content and content != 'N/A':
                        snippet = truncate_text(content, 200)
                        st.markdown(f"*{snippet}*")

                        # Full article in expander
                        with st.expander("📖 Read Full Article"):
                            st.markdown(content)
                    else:
                        st.caption("No content available")

                st.markdown("---")

            # Prediction history
            st.subheader("📊 Recent Prediction History")
            history = load_prediction_history(limit=10)
            if history:
                history_fig = plot_prediction_history(history)
                if history_fig:
                    st.plotly_chart(history_fig, use_container_width=True)
            else:
                st.info("No prediction history available yet. Predictions will be tracked automatically.")

            # Event-driven status
            st.success("✅ **Event-Driven Mode Active**: Predictions automatically generated when new news arrives")
            st.info("💡 **How it works**: The realtime predictor monitors the news directory and triggers predictions using real market features (RSI, MACD, Bollinger Bands, etc.) combined with news sentiment")

        else:
            st.warning("⚠️ No predictions available yet")
            st.info("""
            **To generate predictions:**
            1. Start the realtime predictor: `python src_clean/ui/realtime_predictor.py`
            2. Simulate news articles: `curl -X POST http://localhost:5001/api/stream/positive`
            3. Predictions will appear here automatically
            """)

        st.markdown("---")
        st.info("💡 **Note**: Predictions are for educational purposes. Always do your own research.")

        # Auto-refresh logic for predictions tab
        if auto_refresh_predictions:
            time.sleep(10)
            st.rerun()

    with tab3:
        st.header("📰 News & Sentiment Analysis")

        # Load recent news
        recent_news = load_recent_news(limit=20)

        if recent_news:
            # News sentiment timeline
            st.subheader("📈 Sentiment Timeline")
            sentiment_fig = plot_news_sentiment_timeline(recent_news)
            if sentiment_fig:
                st.plotly_chart(sentiment_fig, use_container_width=True)

            st.markdown("---")

            # Recent news articles
            st.subheader("📄 Recent News Articles")

            # Filter options
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                sentiment_filter = st.selectbox(
                    "Filter by Sentiment",
                    ["All", "Positive", "Negative", "Neutral"]
                )
            with col_filter2:
                num_articles = st.slider("Number of articles", 5, 20, 10)

            # Apply filters
            filtered_news = recent_news[:num_articles]
            if sentiment_filter != "All":
                filtered_news = [
                    n for n in filtered_news
                    if n.get('sentiment_type', '').lower() == sentiment_filter.lower()
                ]

            # Display articles as cards
            for i, article in enumerate(filtered_news):
                sentiment_type = article.get('sentiment_type', 'neutral')
                sentiment_score = article.get('sentiment_score', 0)

                if sentiment_type == 'positive':
                    sentiment_emoji = "🟢"
                elif sentiment_type == 'negative':
                    sentiment_emoji = "🔴"
                else:
                    sentiment_emoji = "⚪"

                with st.expander(f"{sentiment_emoji} {article.get('headline', 'N/A')}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Source:** {article.get('source', 'N/A').replace('_', ' ').title()}")
                        st.write(f"**Published:** {article.get('published_at', 'N/A')[:10]}")
                    with col_b:
                        st.write(f"**Sentiment:** {sentiment_type.title()} ({sentiment_score:.2f})")
                        st.write(f"**Time:** {article['file_time'].strftime('%H:%M:%S')}")

                    # Content snippet
                    content = article.get('content', 'N/A')
                    if content and content != 'N/A':
                        st.markdown("**Preview:**")
                        st.markdown(f"*{truncate_text(content, 300)}*")
                    else:
                        st.caption("No content available")

            # Summary stats
            st.markdown("---")
            st.subheader("📊 News Summary")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)

            sentiments = [a.get('sentiment_score', 0) for a in recent_news]
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            positive_count = sum(1 for a in recent_news if a.get('sentiment_type') == 'positive')
            negative_count = sum(1 for a in recent_news if a.get('sentiment_type') == 'negative')
            neutral_count = sum(1 for a in recent_news if a.get('sentiment_type') == 'neutral')

            with col_s1:
                st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
            with col_s2:
                st.metric("🟢 Positive", positive_count)
            with col_s3:
                st.metric("🔴 Negative", negative_count)
            with col_s4:
                st.metric("⚪ Neutral", neutral_count)

        else:
            st.info("No news articles available. Start the news simulator to generate articles.")

    with tab4:
        st.header("📊 Model Performance Metrics")

        if metrics:
            model_type = metrics.get('model_type', 'unknown')

            # Display basic model info
            st.info(f"**Model Type:** {model_type.title()} | **Features:** {metrics.get('n_features', 0)}")

            # Display training metrics
            st.subheader("📈 Training Metrics")
            display_model_metrics(metrics)

            st.markdown("---")

            # Compute real-time performance from prediction logs
            st.subheader("🔄 Real-Time Performance (from Prediction Logs)")

            try:
                from pathlib import Path
                import json

                pred_log_file = Path("data_clean/predictions/prediction_log.jsonl")

                if pred_log_file.exists() and pred_log_file.stat().st_size > 0:
                    # Load recent predictions
                    predictions = []
                    with open(pred_log_file, 'r') as f:
                        for line in f:
                            try:
                                predictions.append(json.loads(line))
                            except:
                                pass

                    if predictions:
                        # Show statistics from prediction logs
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Predictions", len(predictions))

                        with col2:
                            bullish_count = sum(1 for p in predictions if p.get('prediction') == 'bullish')
                            st.metric("Bullish Predictions", bullish_count)

                        with col3:
                            bearish_count = sum(1 for p in predictions if p.get('prediction') == 'bearish')
                            st.metric("Bearish Predictions", bearish_count)

                        with col4:
                            avg_confidence = np.mean([p.get('confidence', 0) for p in predictions])
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")

                        # Plot prediction distribution over time
                        st.markdown("#### Prediction Timeline")

                        pred_df = pd.DataFrame(predictions)
                        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])

                        fig = px.scatter(
                            pred_df,
                            x='timestamp',
                            y='predicted_relative_change',
                            color='prediction',
                            size='confidence',
                            hover_data=['predicted_price', 'confidence', 'features_used'],
                            title="Predicted Price Changes Over Time",
                            color_discrete_map={'bullish': 'green', 'bearish': 'red'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        # Recent predictions table
                        with st.expander("📋 Recent Predictions"):
                            recent_preds = pred_df.tail(20)[['timestamp', 'prediction', 'confidence',
                                                               'predicted_price', 'predicted_relative_change',
                                                               'features_used']].copy()
                            recent_preds['predicted_relative_change'] = recent_preds['predicted_relative_change'].apply(lambda x: f"{x*100:.2f}%")
                            recent_preds['confidence'] = recent_preds['confidence'].apply(lambda x: f"{x:.2%}")
                            st.dataframe(recent_preds, use_container_width=True)
                    else:
                        st.info("No predictions logged yet.")
                else:
                    st.info("No prediction log file found. Predictions will be logged as they are made.")

            except Exception as e:
                st.error(f"Error loading prediction logs: {e}")

            # Show confusion matrix only for classification models
            if model_type == 'classification' and 'confusion_matrix' in metrics:
                st.markdown("---")
                st.subheader("🎯 Confusion Matrix")
                cm = np.array(metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['DOWN', 'UP'],
                    y=['DOWN', 'UP'],
                    text_auto=True,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Classification report
                with st.expander("📋 Detailed Classification Report"):
                    st.json(metrics.get('classification_report', {}))

        else:
            st.warning("⚠️ No model metrics available. Model may not have associated metrics file.")
            st.info("The system is using the trained XGBoost regression model for predictions.")

    with tab5:
        st.header("🔍 Feature Analysis")

        # Convert feature_names to list first to avoid numpy array boolean ambiguity in conditional
        if model is not None and feature_names is not None:
            # Ensure feature_names is a Python list
            if hasattr(feature_names, 'tolist'):
                feature_names_list = feature_names.tolist()
            elif hasattr(feature_names, '__iter__') and not isinstance(feature_names, str):
                feature_names_list = list(feature_names)
            else:
                feature_names_list = feature_names

            if feature_names_list:
                st.info(f"**Model has {len(feature_names_list)} features**")

                # Try to plot feature importance
                fig = plot_feature_importance(model, feature_names_list)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Feature importance plot not available for this model type")
                    st.info("Feature importance requires a tree-based model (e.g., XGBoost, Random Forest)")

                # Feature list grouped by category
                st.subheader("📋 Feature Categories")

                feature_categories = {
                    "OHLCV": ["open", "high", "low", "close", "volume"],
                    "Returns": ["return_1", "return_5", "return_10"],
                    "RSI": ["rsi_14", "rsi_20"],
                    "MACD": ["macd", "macd_signal", "macd_histogram"],
                    "Bollinger Bands": ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position"],
                    "Moving Averages": ["sma_7", "sma_14", "sma_21", "sma_50", "ema_7", "ema_14", "ema_21"],
                    "Volatility": ["atr_14", "adx_14", "volatility_20", "volatility_50", "hist_vol_20", "hist_vol_50"],
                    "Advanced Volatility": ["gk_vol", "parkinson_vol", "rs_vol", "yz_vol", "vol_of_vol", "ewma_vol"],
                    "Momentum": ["momentum_5", "momentum_10", "momentum_20", "roc_5", "roc_10"],
                    "Range Indicators": ["hl_range", "hl_range_pct", "hl_range_ma20", "realized_range", "realized_range_ma"],
                    "Volume": ["volume_ma20", "volume_ma50", "volume_ratio", "volume_zscore", "volume_velocity", "volume_acceleration"],
                    "Microstructure": ["price_impact", "price_impact_ma20", "order_flow_imbalance", "illiquidity", "illiquidity_ma20"],
                    "VWAP": ["vwap", "close_vwap_ratio"],
                    "Spread": ["spread_proxy", "spread_pct"],
                    "Volatility Regime": ["vol_regime_low", "vol_regime_high"],
                    "News Sentiment": [f for f in feature_names_list if 'news' in f.lower()]
                }

                # Display features by category
                col1, col2 = st.columns(2)
                categories_list = list(feature_categories.items())
                mid = len(categories_list) // 2

                with col1:
                    for category, features in categories_list[:mid]:
                        with st.expander(f"{category} ({len([f for f in features if f in feature_names_list])})"):
                            matching_features = [f for f in features if f in feature_names_list]
                            if matching_features:
                                for feat in matching_features:
                                    st.text(f"• {feat}")
                            else:
                                st.caption("No matching features")

                with col2:
                    for category, features in categories_list[mid:]:
                        with st.expander(f"{category} ({len([f for f in features if f in feature_names_list])})"):
                            matching_features = [f for f in features if f in feature_names_list]
                            if matching_features:
                                for feat in matching_features:
                                    st.text(f"• {feat}")
                            else:
                                st.caption("No matching features")

                # All features list
                with st.expander("📋 All Features (Raw List)"):
                    st.text("\n".join(feature_names_list))
        else:
            st.warning("⚠️ No model or feature data available")
            st.info("The Feature Analysis tab requires a local model file to display feature importance and details.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit | Data from OANDA | ML Model: XGBoost</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
