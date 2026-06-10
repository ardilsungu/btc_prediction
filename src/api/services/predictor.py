"""
Predictor — Applies feature engineering, generates model predictions.

Separate prediction functions for both model types (LR and XGBoost).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Add src/ path to import feature_engineering.py
_SRC_DIR = Path(__file__).resolve().parents[2]  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from feature_engineering import (  # noqa: E402
    TIMEFRAME_CONFIGS,
    engineer_features,
    get_feature_columns,
)
from api.services.model_loader import LRModelBundle, XGBModelBundle  # noqa: E402

logger = logging.getLogger(__name__)

# Prediction label map
LABEL_MAP = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}


# ──────────────────────────────────────────────────────────────
# HELPER: Feature Preparation
# ──────────────────────────────────────────────────────────────
def _prepare_features(df: pd.DataFrame, timeframe: str) -> Tuple[np.ndarray, list, pd.Timestamp]:
    """
    Generates feature matrix from Binance DataFrame.

    Returns
    -------
    X_last   : shape (1, n_features) — feature vector of the most recent candle
    feat_cols: list of feature names
    candle_ts: timestamp of the last candle
    """
    cfg = TIMEFRAME_CONFIGS[timeframe]

    # feature_engineering.py uses the "Open time" column with dt accessor
    df = df.copy()
    df = engineer_features(df, cfg)

    feat_cols = get_feature_columns(cfg)

    # clean inf / NaN
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)

    # Get the last clean row
    df_clean = df.dropna(subset=feat_cols)
    if df_clean.empty:
        raise ValueError("No clean rows left after feature engineering.")

    last_row   = df_clean.iloc[-1]
    candle_ts  = last_row["Open time"]
    X_last     = last_row[feat_cols].values.reshape(1, -1)

    return X_last, feat_cols, candle_ts


# ──────────────────────────────────────────────────────────────
# LR PREDICTION
# ──────────────────────────────────────────────────────────────
def predict_lr(bundle: LRModelBundle, df: pd.DataFrame, timeframe: str) -> dict:
    """
    Generates prediction with Logistic Regression.

    Returns
    -------
    dict with the following fields:
        prediction     : "UP" | "DOWN" | "NEUTRAL"
        confidence     : {up, down, neutral}  (float, 0-1)
        current_price  : float
        candle_time    : pd.Timestamp
        model_metrics  : dict
    """
    X_last, _, candle_ts = _prepare_features(df, timeframe)

    # Scale
    X_scaled = bundle.scaler.transform(X_last)

    # Probabilities — sklearn LogisticRegression.classes_ = [-1, 0, 1]
    proba = bundle.model.predict_proba(X_scaled)[0]
    classes = bundle.model.classes_  # [-1, 0, 1]

    prob_map = {int(c): float(p) for c, p in zip(classes, proba)}
    p_down    = prob_map.get(-1, 0.0)
    p_neutral = prob_map.get(0,  0.0)
    p_up      = prob_map.get(1,  0.0)

    # Class with the highest probability
    pred_class = int(classes[np.argmax(proba)])
    prediction = LABEL_MAP[pred_class]

    current_price = float(df["Close"].iloc[-1])

    logger.info(
        "[LR][%s] Prediction: %s | UP=%.2f DOWN=%.2f NEUTRAL=%.2f | Price: %.2f",
        timeframe, prediction, p_up, p_down, p_neutral, current_price,
    )

    return {
        "prediction":    prediction,
        "confidence":    {"up": p_up, "down": p_down, "neutral": p_neutral},
        "current_price": current_price,
        "candle_time":   candle_ts,
        "threshold_used": None,
        "model_metrics": bundle.metrics or None,
    }


# ──────────────────────────────────────────────────────────────
# XGBOOST PREDICTION (Binary Relevance)
# ──────────────────────────────────────────────────────────────
def predict_xgb(bundle: XGBModelBundle, df: pd.DataFrame, timeframe: str) -> dict:
    """
    Generates prediction with XGBoost Binary Relevance.

    model_up  → P(UP)   — compared with thresh_up
    model_down→ P(DOWN) — compared with thresh_down

    Returns
    -------
    dict (same structure as predict_lr + threshold_used)
    """
    X_last, _, candle_ts = _prepare_features(df, timeframe)

    # Binary Relevance probabilities
    p_up_raw   = float(bundle.model_up.predict_proba(X_last)[0, 1])
    p_down_raw = float(bundle.model_down.predict_proba(X_last)[0, 1])

    thresh_up   = bundle.thresh_up
    thresh_down = bundle.thresh_down

    # Decision logic (same as in training)
    above_up   = p_up_raw   >= thresh_up
    above_down = p_down_raw >= thresh_down

    if above_up and not above_down:
        prediction = "UP"
    elif above_down and not above_up:
        prediction = "DOWN"
    elif above_up and above_down:
        # Overlap: choose the one with higher confidence
        prediction = "UP" if p_up_raw > p_down_raw else "DOWN"
    else:
        prediction = "NEUTRAL"

    # Normalize confidence (between 0-1, sum ≈ 1)
    p_neutral_raw = max(0.0, 1.0 - p_up_raw - p_down_raw)
    total         = p_up_raw + p_down_raw + p_neutral_raw + 1e-9
    p_up      = p_up_raw   / total
    p_down    = p_down_raw / total
    p_neutral = p_neutral_raw / total

    current_price = float(df["Close"].iloc[-1])

    logger.info(
        "[XGB][%s] Prediction: %s | UP=%.3f(thr=%.3f) DOWN=%.3f(thr=%.3f) | Price: %.2f",
        timeframe, prediction,
        p_up_raw, thresh_up,
        p_down_raw, thresh_down,
        current_price,
    )

    return {
        "prediction":    prediction,
        "confidence":    {"up": p_up, "down": p_down, "neutral": p_neutral},
        "current_price": current_price,
        "candle_time":   candle_ts,
        "threshold_used": {"up": thresh_up, "down": thresh_down},
        "model_metrics": None,
    }
