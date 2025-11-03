import logging
import sys
from typing import Dict, Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings


class LoggingMiddleware(BaseHTTPMiddleware):
    """日志中间件"""
    
    async def dispatch(self, request: Request, call_next):
        # 记录请求开始
        logger = logging.getLogger(__name__)
        logger.info(
            f"请求开始: {request.method} {request.url.path} - "
            f"客户端: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            
            # 记录请求完成
            logger.info(
                f"请求完成: {request.method} {request.url.path} - "
                f"状态码: {response.status_code}"
            )
            
            return response
            
        except Exception as e:
            # 记录异常
            logger.error(
                f"请求异常: {request.method} {request.url.path} - "
                f"错误信息: {str(e)}"
            )
            raise


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """异常处理中间件"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
            
        except ValueError as e:
            # 参数验证错误
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "请求参数错误",
                    "detail": str(e),
                    "code": "VALIDATION_ERROR"
                }
            )
            
        except KeyError as e:
            # 缺少必要参数
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "缺少必要参数",
                    "detail": f"缺少参数: {str(e)}",
                    "code": "MISSING_PARAMETER"
                }
            )
            
        except Exception as e:
            # 服务器内部错误
            logger = logging.getLogger(__name__)
            logger.error(f"服务器内部错误: {str(e)}", exc_info=True)
            
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "服务器内部错误",
                    "detail": str(e) if settings.debug else "请联系管理员",
                    "code": "INTERNAL_SERVER_ERROR"
                }
            )


def setup_logging():
    """配置日志系统"""
    # 创建根日志记录器
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # 设置第三方库日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)