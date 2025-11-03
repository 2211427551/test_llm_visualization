from typing import AsyncGenerator

from fastapi import Depends

from app.core.config import settings


async def get_settings():
    """获取应用配置的依赖注入函数"""
    return settings


class CommonDependencies:
    """通用依赖注入类"""
    
    def __init__(self):
        self.settings = settings
    
    @classmethod
    def get_settings(cls):
        """获取应用配置"""
        return settings


# 创建通用依赖实例
common_deps = CommonDependencies()