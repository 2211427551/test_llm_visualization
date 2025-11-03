from datetime import datetime
from typing import Optional, Any, Dict

from pydantic import BaseModel


class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = True
    message: str = "操作成功"
    data: Optional[Any] = None
    timestamp: datetime = datetime.now()


class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = False
    message: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: datetime = datetime.now()


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = "healthy"
    version: str
    timestamp: datetime = datetime.now()
    services: Dict[str, str] = {}