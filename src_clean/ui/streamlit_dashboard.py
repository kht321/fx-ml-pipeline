"""
Enhanced Streamlit Dashboard for S&P500 ML Prediction Pipeline

Features:
- Real-time OANDA price data
- ML predictions with confidence scores
- Feature importance visualization
- Model metrics dashboard
- News sentiment integration
- Historical prediction performance
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

# Try to import OANDA (optional - fallback to mock data)
try:
    from oanda_api import API
    import oandapyV20.endpoints.pricing as pricing
    import oandapyV20.endpoints.instruments as instruments
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    st.warning("OANDA API not available - using mock data")

# Configuration
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_TOKEN = os.getenv("OANDA_TOKEN")
MODEL_DIR = Path("data_clean/models")
GOLD_DIR = Path("data_clean/gold")
PREDICTIONS_FILE = Path("data_clean/predictions/latest_prediction.json")


@st.cache_resource
def load_latest_model():
    """Load the most recent trained model."""
    try:
        # Find latest model
        model_files = list(MODEL_DIR.glob("xgboost_classification_*.pkl"))
        if not model_files:
            return None, None, None

        latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        model = load(latest_model_path)

        # Load features and metrics
        model_name = latest_model_path.stem
        features_path = MODEL_DIR / f"{model_name}_features.json"
        metrics_path = MODEL_DIR / f"{model_name}_metrics.json"

        with open(features_path) as f:
            features = json.load(f)['features']

        with open(metrics_path) as f:
            metrics = json.load(f)

        return model, features, metrics
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def get_latest_price(account_id, instrument="SPX500_USD"):
    """Get latest price from OANDA."""
    if not OANDA_AVAILABLE or not account_id:
        # Mock data
        return 4521.50 + np.random.randn() * 10

    try:
        params = {"instruments": instrument}
        request = pricing.PricingInfo(accountID=account_id, params=params)
        response = API.request(request)
        prices = response['prices'][0]
        bid = float(prices['bids'][0]['price'])
        ask = float(prices['asks'][0]['price'])
        return (bid + ask) / 2
    except Exception as e:
        st.error(f"Error fetching price: {e}")
        return None


def get_candles(instrument="SPX500_USD", granularity="H1", count=100):
    """Get historical candles from OANDA."""
    if not OANDA_AVAILABLE:
        # Generate mock candles
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

    try:
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M"
        }
        request = instruments.InstrumentsCandles(instrument=instrument, params=params)
        response = API.request(request)
        time.sleep(0.5)
        candles = response.get("candles", [])

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
        return None


def calculate_technical_features(df):
    """Calculate basic technical features for prediction."""
    # Simple features for demo
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    return df


def load_latest_prediction():
    """Load the latest prediction from event-driven predictor."""
    try:
        if not PREDICTIONS_FILE.exists():
            return None

        with open(PREDICTIONS_FILE, 'r') as f:
            pred_data = json.load(f)

        # Convert to display format
        return {
            'prediction': pred_data['prediction'],
            'confidence': pred_data['confidence'],
            'prob_up': pred_data['probabilities']['UP'],
            'prob_down': pred_data['probabilities']['DOWN'],
            'timestamp': pred_data['timestamp'],
            'trigger': pred_data.get('trigger', 'unknown'),
            'features_used': pred_data.get('features_calculated', 0)
        }
    except Exception as e:
        st.error(f"Error loading prediction: {e}")
        return None


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


def plot_candlestick(df):
    """Create candlestick chart with volume."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('S&P 500 Price', 'Volume'),
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
        showlegend=False,
        title_text="S&P 500 Market Data"
    )

    return fig


def plot_feature_importance(model, feature_names):
    """Plot feature importance from model."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 20 Feature Importance'
    )
    fig.update_layout(height=600)

    return fig


def display_model_metrics(metrics):
    """Display model performance metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with col2:
        st.metric("AUC", f"{metrics.get('auc', 0):.4f}")
    with col3:
        st.metric("CV Mean", f"{metrics.get('cv_mean', 0):.4f}")
    with col4:
        st.metric("CV Std", f"{metrics.get('cv_std', 0):.4f}")


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
    refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 30, 300, 60)

    # Load model
    st.sidebar.header("🤖 Model Status")
    model, feature_names, metrics = load_latest_model()

    if model is None:
        st.sidebar.error("❌ No model loaded")
        st.error("No trained model found. Please train a model first.")
        return
    else:
        st.sidebar.success("✅ Model loaded")
        st.sidebar.info(f"Features: {len(feature_names)}")

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Trading", "🎯 Predictions", "📊 Model Performance", "🔍 Feature Analysis"])

    with tab1:
        st.header(f"Live Market Data: {instrument}")

        # Current price
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            current_price = get_latest_price(OANDA_ACCOUNT_ID, instrument)
            if current_price:
                st.metric("Current Price", f"${current_price:.2f}", delta=None)

        # Fetch and display candles
        df = get_candles(instrument, granularity, candle_count)
        if df is not None and not df.empty:
            # Plot
            fig = plot_candlestick(df)
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
            auto_refresh_predictions = st.checkbox("🔄 Auto-refresh predictions every 5 seconds", value=False)

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

            # Display news article that triggered prediction
            if 'news_article' in pred_result and pred_result['news_article']:
                st.subheader("📰 News Article That Triggered This Prediction")
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

                # Display news in a nice card
                st.markdown(f"### {sentiment_color} {news.get('headline', 'N/A')}")

                col_source, col_time = st.columns(2)
                with col_source:
                    st.markdown(f"**Source:** {news.get('source', 'N/A').replace('_', ' ').title()}")
                with col_time:
                    st.markdown(f"**Published:** {news.get('published_at', 'N/A')[:10]}")

                st.markdown(f"**Sentiment:** {sentiment_badge}")

                # Show content in expander
                if news.get('content') and news.get('content') != 'N/A':
                    with st.expander("📖 Read Full Article"):
                        st.markdown(news['content'])

                st.markdown("---")

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
            time.sleep(5)
            st.rerun()

    with tab3:
        st.header("📊 Model Performance Metrics")

        if metrics:
            display_model_metrics(metrics)

            st.markdown("### Confusion Matrix")
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
            st.warning("No metrics available")

    with tab4:
        st.header("🔍 Feature Analysis")

        if model and feature_names:
            fig = plot_feature_importance(model, feature_names)
            st.plotly_chart(fig, use_container_width=True)

            # Feature list
            with st.expander("📋 All Features"):
                st.write(feature_names)
        else:
            st.warning("No feature data available")

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

    # Auto-refresh
    if st.sidebar.checkbox("🔄 Auto-refresh", value=False):
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
