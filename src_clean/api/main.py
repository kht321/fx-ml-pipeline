"""FastAPI main application for S&P 500 ML prediction API."""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    HistoricalPredictionsResponse,
    NewsArticle
)
from .inference import ModelInference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="S&P 500 ML Prediction API",
    description="Real-time ML predictions for S&P 500 using market data and news sentiment",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model inference engine
try:
    model = ModelInference(
        model_path="models/gradient_boosting_combined_model.pkl",
        feast_repo="feature_repo"
    )
    logger.info("Model inference engine initialized")
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    model = None


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for market streaming."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)


ws_manager = ConnectionManager()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "S&P 500 ML Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "news": "/news/recent",
            "websocket": "/ws/market-stream"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    redis_connected = False
    feast_available = False

    # Check Redis connection
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
        r.ping()
        redis_connected = True
    except Exception:
        pass

    # Check Feast availability
    if model and model.feast_store:
        feast_available = True

    return HealthResponse(
        status="healthy" if model and model.is_loaded else "degraded",
        timestamp=datetime.now(),
        model_loaded=model.is_loaded if model else False,
        feast_available=feast_available,
        redis_connected=redis_connected
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """Generate ML prediction for S&P 500.

    Args:
        request: Prediction request with instrument and optional timestamp

    Returns:
        PredictionResponse with prediction details
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        prediction = model.predict(
            instrument=request.instrument,
            timestamp=request.timestamp
        )

        return PredictionResponse(**prediction)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predictions/history", response_model=HistoricalPredictionsResponse, tags=["Predictions"])
async def get_prediction_history(
    instrument: str = "SPX500_USD",
    hours: int = 24
):
    """Get historical predictions.

    Args:
        instrument: Trading instrument
        hours: Number of hours to look back

    Returns:
        Historical predictions
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    # For now, generate mock historical data
    # In production, this would read from a predictions database
    predictions = []

    current_time = start_time
    while current_time < end_time:
        pred = model.predict(instrument, current_time) if model else {}
        predictions.append({
            "time": current_time.isoformat(),
            "prediction": pred.get("prediction", "neutral"),
            "probability": pred.get("probability", 0.5),
            "confidence": pred.get("confidence", 0.5)
        })
        current_time += timedelta(hours=1)

    return HistoricalPredictionsResponse(
        instrument=instrument,
        start_time=start_time,
        end_time=end_time,
        predictions=predictions,
        count=len(predictions)
    )


@app.get("/news/recent", response_model=List[NewsArticle], tags=["News"])
async def get_recent_news(limit: int = 10):
    """Get recent news articles with sentiment analysis.

    Args:
        limit: Maximum number of articles to return

    Returns:
        List of news articles
    """
    if not model:
        return []

    try:
        news = await model.get_recent_news(limit=limit)
        return [NewsArticle(**article) for article in news]

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []


@app.websocket("/ws/market-stream")
async def market_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time market data and predictions.

    Streams:
        - Latest market price
        - ML predictions
        - Confidence scores
        - News sentiment

    Update interval: 5 seconds
    """
    await ws_manager.connect(websocket)

    try:
        while True:
            if model:
                # Get latest prediction
                prediction = await model.get_latest_prediction()

                # Add market data (mock for now)
                market_update = {
                    "type": "market_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "instrument": "SPX500_USD",
                        "price": 4521.50,  # Mock - would fetch from OANDA
                        "volume": 1234567,
                        "prediction": prediction.get("prediction"),
                        "probability": prediction.get("probability"),
                        "confidence": prediction.get("confidence"),
                        "signal_strength": prediction.get("signal_strength")
                    }
                }

                await websocket.send_json(market_update)

            # Wait before next update (5 seconds)
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("Client disconnected from market stream")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting S&P 500 ML Prediction API")
    logger.info(f"Model loaded: {model.is_loaded if model else False}")

    # Start background task for predictions if needed
    # asyncio.create_task(background_prediction_updater())


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down S&P 500 ML Prediction API")


# Development/testing endpoint
@app.get("/debug/model-info", tags=["Debug"])
async def model_info():
    """Get model information (for debugging)."""
    if not model:
        return {"error": "Model not loaded"}

    return {
        "model_loaded": model.is_loaded,
        "model_type": model.model_type,
        "features_count": len(model.feature_names) if model.feature_names else 0,
        "feast_available": model.feast_store is not None,
        "model_path": str(model.model_path)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
