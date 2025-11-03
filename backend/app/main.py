from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.middleware import LoggingMiddleware, ExceptionHandlerMiddleware, setup_logging
from app.routers import api_router


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    
    # 配置日志
    setup_logging()
    
    # 创建 FastAPI 应用
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="基于 FastAPI 的后端应用",
        debug=settings.debug,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # 添加中间件
    app.add_middleware(ExceptionHandlerMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # 添加 CORS 中间件（根据需要配置）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境中应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(api_router, prefix=settings.api_prefix)
    
    # 根路径重定向到健康检查
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "欢迎使用 FastAPI 后端应用", "docs": "/docs", "health": "/api/v1/health"}
    
    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )