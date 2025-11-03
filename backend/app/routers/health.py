from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.schemas.common import HealthResponse
from app.services.health import HealthService


router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check() -> HealthResponse:
    """
    检查应用健康状态
    
    返回应用的基本健康信息，包括版本号、时间戳等。
    """
    return await HealthService.check_health()


@router.get("/info", summary="应用信息")
async def app_info() -> Dict[str, str]:
    """
    获取应用基本信息
    
    返回应用的名称、版本等基本信息。
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "debug": str(settings.debug),
        "timestamp": datetime.now().isoformat()
    }