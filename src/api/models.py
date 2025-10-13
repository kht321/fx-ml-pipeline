"""Pydantic models for API request/response validation."""

from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    instrument: str = Field(default="SPX500_USD", description="Trading instrument")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp for historical prediction")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    instrument: str
    timestamp: datetime
    prediction: str  # "bullish", "bearish", "neutral"
    probability: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    signal_strength: float  # -1.0 to 1.0
    features_used: int
    model_version: str


class MarketDataPoint(BaseModel):
    """Single market data point."""
    time: datetime
    price: float
    volume: Optional[float] = None
    prediction: Optional[str] = None
    probability: Optional[float] = None


class NewsArticle(BaseModel):
    """News article with sentiment."""
    time: datetime
    headline: str
    source: str
    sentiment: float  # -1.0 to 1.0
    impact: str  # "low", "medium", "high"
    url: Optional[str] = None


class HistoricalPredictionsResponse(BaseModel):
    """Response for historical predictions."""
    instrument: str
    start_time: datetime
    end_time: datetime
    predictions: List[Dict]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    feast_available: bool
    redis_connected: bool
