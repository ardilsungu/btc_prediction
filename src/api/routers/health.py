"""
/health endpoint
"""
from fastapi import APIRouter
from api.models import HealthResponse, LoadedModels
from api.services.model_loader import get_registry

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health status",
    description="Returns whether the service is up and which models are loaded.",
)
async def health_check() -> HealthResponse:
    registry = get_registry()

    xgb_tfs = sorted(registry.loaded_xgb_timeframes())
    lr_tfs   = sorted(registry.loaded_lr_timeframes())

    total = len(xgb_tfs) + len(lr_tfs)
    status = "healthy" if total > 0 else "degraded"

    return HealthResponse(
        status=status,
        loaded_models=LoadedModels(xgboost=xgb_tfs, lr=lr_tfs),
        message=f"Total of {total} models loaded. BTC Prediction API is running.",
    )
