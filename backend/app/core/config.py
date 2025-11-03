import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    # 应用基础配置
    app_name: str = Field(default="FastAPI 后端应用", description="应用名称")
    app_version: str = Field(default="0.1.0", description="应用版本")
    debug: bool = Field(default=True, description="调试模式")
    
    # 服务器配置
    host: str = Field(default="0.0.0.0", description="服务器地址")
    port: int = Field(default=8000, description="服务器端口")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    
    # API 配置
    api_prefix: str = Field(default="/api/v1", description="API前缀")


@lru_cache()
def get_settings() -> Settings:
    """获取应用配置实例（单例模式）"""
    return Settings()


# 全局配置实例
settings = get_settings()