from fastapi import APIRouter

from app.routers import health

# 创建主路由器
api_router = APIRouter()

# 注册各个模块的路由
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["健康检查"]
)