"""
Model Loader — Loads LR and XGBoost models from disk,
loads them into RAM at application startup, and keeps them in cache.

Directory structure:
  data/LRmodels/{timeframe}/
      model.joblib
      scaler.joblib
      metrics.json

  data/XGmodels/{timeframe}/
      model_up.json
      model_down.json
      meta.json   ← thresh_up, thresh_down
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import xgboost as xgb

logger = logging.getLogger(__name__)

# Project root: src/api/services/ → src/api/ → src/ → project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LR_DIR  = PROJECT_ROOT / "data" / "LRmodels"
XGB_DIR = PROJECT_ROOT / "data" / "XGmodels"

TIMEFRAMES = ["15m", "1h", "4h", "1d"]


# ──────────────────────────────────────────────────────────────
# DATA CLASS: LR Model Bundle
# ──────────────────────────────────────────────────────────────
@dataclass
class LRModelBundle:
    timeframe: str
    model: Any          # sklearn LogisticRegression
    scaler: Any         # sklearn StandardScaler
    metrics: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# DATA CLASS: XGBoost Model Bundle
# ──────────────────────────────────────────────────────────────
@dataclass
class XGBModelBundle:
    timeframe: str
    model_up: xgb.XGBClassifier
    model_down: xgb.XGBClassifier
    thresh_up: float
    thresh_down: float
    metrics: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ──────────────────────────────────────────────────────────────
class ModelRegistry:
    """
    Created once at application startup.
    Keeps all models in memory.
    """

    def __init__(self) -> None:
        self._lr_models:  Dict[str, LRModelBundle]  = {}
        self._xgb_models: Dict[str, XGBModelBundle] = {}
        self._load_all()

    # ── Loading ──────────────────────────────────────────────

    def _load_all(self) -> None:
        for tf in TIMEFRAMES:
            self._try_load_lr(tf)
            self._try_load_xgb(tf)

        logger.info(
            "Model Registry ready — LR: %s | XGB: %s",
            list(self._lr_models.keys()),
            list(self._xgb_models.keys()),
        )

    def _try_load_lr(self, timeframe: str) -> None:
        model_path  = LR_DIR / timeframe / "model.joblib"
        scaler_path = LR_DIR / timeframe / "scaler.joblib"

        if not model_path.exists() or not scaler_path.exists():
            logger.warning("[LR][%s] Model files not found, skipping.", timeframe)
            return

        try:
            model  = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # metrics.json (optional)
            metrics = {}
            metrics_path = LR_DIR / timeframe / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            self._lr_models[timeframe] = LRModelBundle(
                timeframe=timeframe,
                model=model,
                scaler=scaler,
                metrics=metrics,
            )
            logger.info("[LR][%s] Loaded ✓", timeframe)

        except Exception as exc:
            logger.error("[LR][%s] Failed to load: %s", timeframe, exc)

    def _try_load_xgb(self, timeframe: str) -> None:
        up_path   = XGB_DIR / timeframe / "model_up.json"
        down_path = XGB_DIR / timeframe / "model_down.json"
        meta_path = XGB_DIR / timeframe / "meta.json"

        if not all(p.exists() for p in [up_path, down_path, meta_path]):
            logger.warning("[XGB][%s] Model files not found, skipping.", timeframe)
            return

        try:
            model_up = xgb.XGBClassifier()
            model_up.load_model(str(up_path))

            model_down = xgb.XGBClassifier()
            model_down.load_model(str(down_path))

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            thresh_up   = float(meta["thresh_up"])
            thresh_down = float(meta["thresh_down"])

            # Try to extract accuracy line from classification_report.txt (optional)
            metrics: Dict[str, float] = {}
            report_path = XGB_DIR / timeframe / "classification_report.txt"
            if report_path.exists():
                metrics["report_available"] = True

            self._xgb_models[timeframe] = XGBModelBundle(
                timeframe=timeframe,
                model_up=model_up,
                model_down=model_down,
                thresh_up=thresh_up,
                thresh_down=thresh_down,
                metrics=metrics,
            )
            logger.info("[XGB][%s] Loaded ✓ (thresh_up=%.3f, thresh_down=%.3f)",
                        timeframe, thresh_up, thresh_down)

        except Exception as exc:
            logger.error("[XGB][%s] Failed to load: %s", timeframe, exc)

    # ── Getters ──────────────────────────────────────────────

    def get_lr(self, timeframe: str) -> Optional[LRModelBundle]:
        return self._lr_models.get(timeframe)

    def get_xgb(self, timeframe: str) -> Optional[XGBModelBundle]:
        return self._xgb_models.get(timeframe)

    def loaded_lr_timeframes(self) -> List[str]:
        return list(self._lr_models.keys())

    def loaded_xgb_timeframes(self) -> List[str]:
        return list(self._xgb_models.keys())


# ── Singleton (single instance application-wide) ──────────────
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """For FastAPI dependency injection or direct call."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
