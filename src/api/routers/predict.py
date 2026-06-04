"""
/api/v1/predict endpoint

Fetches current OHLCV from Binance, applies feature engineering,
generates and returns prediction with the selected model.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from api.models import Confidence, PredictResponse, ThresholdInfo, ModelMetrics
from api.services.binance_client import fetch_ohlcv
from api.services.model_loader import get_registry
from api.services.predictor import predict_lr, predict_xgb

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Prediction"])

VALID_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
VALID_MODELS     = ["xgboost", "lr"]


@router.get(
    "/predict",
    response_model=PredictResponse,
    summary="Generate BTC price prediction",
    description="""
Fetches current BTC/USDT OHLCV data from Binance, applies feature engineering
and predicts the direction of the next candle using the selected ML model.

**Prediction labels:**
- `UP`      — Price will increase above threshold
- `DOWN`    — Price will decrease below threshold  
- `NEUTRAL` — Price will remain within threshold

**Model differences:**
- `xgboost` — Binary Relevance (separate models for UP + DOWN), threshold-based decision
- `lr`       — 3-class Logistic Regression, class with the highest probability is selected
""",
    responses={
        200: {"description": "Prediction generated successfully"},
        422: {"description": "Invalid parameter"},
        503: {"description": "Requested model not loaded"},
        502: {"description": "Failed to connect to Binance"},
        500: {"description": "Prediction generation error"},
    },
)
async def predict(
    model: Literal["xgboost", "lr"] = Query(
        default="xgboost",
        description="Model to be used: `xgboost` or `lr`",
    ),
    timeframe: Literal["15m", "1h", "4h", "1d"] = Query(
        default="1h",
        description="Timeframe: `15m`, `1h`, `4h`, `1d`",
    ),
) -> PredictResponse:

    registry = get_registry()

    # ── 1. Check if model exists ──────────────────────────
    if model == "xgboost":
        bundle = registry.get_xgb(timeframe)
    else:
        bundle = registry.get_lr(timeframe)

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is not loaded for timeframe '{timeframe}'.",
        )

    # ── 2. Fetch data from Binance ─────────────────────────────
    try:
        df = fetch_ohlcv(timeframe)
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Binance data error")
        raise HTTPException(status_code=502, detail=f"Binance data error: {exc}") from exc

    # ── 3. Generate prediction ──────────────────────────────────────
    try:
        if model == "xgboost":
            result = predict_xgb(bundle, df, timeframe)
        else:
            result = predict_lr(bundle, df, timeframe)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    # ── 4. Create response ─────────────────────────────────────
    threshold_used = None
    if result["threshold_used"]:
        threshold_used = ThresholdInfo(**result["threshold_used"])

    model_metrics = None
    if result.get("model_metrics"):
        m = result["model_metrics"]
        model_metrics = ModelMetrics(
            accuracy=m.get("accuracy"),
            macro_f1=m.get("macro_f1"),
            weighted_f1=m.get("weighted_f1"),
        )

    return PredictResponse(
        timeframe=timeframe,
        model=model,
        current_price=result["current_price"],
        prediction=result["prediction"],
        confidence=Confidence(**result["confidence"]),
        threshold_used=threshold_used,
        model_metrics=model_metrics,
        candle_time=result["candle_time"],
    )
