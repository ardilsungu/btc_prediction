"""
FastAPI Application Entry Point

Startup:
    python run_api.py
    or:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI : http://localhost:8000/docs
ReDoc       : http://localhost:8000/redoc
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add src/ folder to Python path (for feature_engineering.py import)
_SRC_DIR = Path(__file__).resolve().parent.parent  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from api.routers import health, predict  # noqa: E402
from api.services.model_loader import get_registry  # noqa: E402

# ── Logging Setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on application startup."""
    logger.info("=" * 60)
    logger.info("  BTC Prediction API is starting...")
    logger.info("=" * 60)
    registry = get_registry()
    xgb_count = len(registry.loaded_xgb_timeframes())
    lr_count  = len(registry.loaded_lr_timeframes())
    logger.info("  XGBoost models : %d / 4", xgb_count)
    logger.info("  LR models      : %d / 4", lr_count)
    logger.info("  Swagger UI     : http://localhost:8000/docs")
    logger.info("=" * 60)
    yield
    logger.info("BTC Prediction API is shutting down.")


# ── FastAPI Application ──────────────────────────────────────────
app = FastAPI(
    title="BTC Prediction API",
    description="""
## Bitcoin Price Prediction Service

Predicts price direction using trained **XGBoost** and **Logistic Regression** models
on recent BTC/USDT data from Binance.

### How it Works
1. Fetch current OHLCV (Open/High/Low/Close/Volume) data from Binance
2. Apply the same feature engineering pipeline used during training
3. The selected model generates a prediction
4. Prediction + confidence scores are returned

### Models
| Model | Architecture | Timeframe |
|-------|--------------|-----------|
| `xgboost` | Binary Relevance (separate UP + DOWN models) | 15m, 1h, 4h, 1d |
| `lr` | 3-class Logistic Regression | 15m, 1h, 4h, 1d |
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — Allow frontend to connect from anywhere ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Provide frontend URL in production
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── Router Registration ─────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)


# ── Root ────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "BTC Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/api/v1/predict?model=xgboost&timeframe=1h",
    }
