"""
LLM 高级可视化后端的 FastAPI 应用程序
"""

from fastapi import FastAPI

app = FastAPI(
    title="LLM 高级可视化 API",
    description="用于 LLM 可视化和分析的后端 API",
    version="0.1.0"
)


@app.get("/")
async def root():
    """根端点"""
    return {"message": "LLM 高级可视化 API 正在运行"}


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "健康"}