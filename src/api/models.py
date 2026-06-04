"""
Pydantic request/response schemas.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# COMMON
# ──────────────────────────────────────────────────────────────
class ModelMetrics(BaseModel):
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None


class ThresholdInfo(BaseModel):
    """Only applicable for XGBoost (Binary Relevance)."""
    up: Optional[float] = None
    down: Optional[float] = None


class Confidence(BaseModel):
    """Probability distribution for both models."""
    up: float = Field(..., ge=0.0, le=1.0, description="UP probability")
    down: float = Field(..., ge=0.0, le=1.0, description="DOWN probability")
    neutral: float = Field(..., ge=0.0, le=1.0, description="NEUTRAL probability")


# ──────────────────────────────────────────────────────────────
# PREDICT RESPONSE
# ──────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    symbol: str = Field(default="BTC/USDT")
    timeframe: str
    model: Literal["xgboost", "lr"]
    current_price: float = Field(..., description="Last candle close price (USDT)")
    prediction: Literal["UP", "DOWN", "NEUTRAL"]
    confidence: Confidence
    threshold_used: Optional[ThresholdInfo] = Field(
        default=None,
        description="Only for XGBoost Binary Relevance"
    )
    model_metrics: Optional[ModelMetrics] = None
    candle_time: datetime = Field(..., description="Candle time when the prediction was made (UTC)")
    requested_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────
# HEALTH RESPONSE
# ──────────────────────────────────────────────────────────────
class LoadedModels(BaseModel):
    xgboost: list[str] = Field(default_factory=list)
    lr: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    loaded_models: LoadedModels
    message: str = "BTC Prediction API is running."


# ──────────────────────────────────────────────────────────────
# ERROR RESPONSE
# ──────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    detail: str
