from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class InitializeResponse(BaseModel):
    """模型初始化响应"""
    success: bool = Field(..., description="初始化是否成功")
    message: str = Field(..., description="响应消息")
    config: Dict[str, Any] = Field(..., description="模型配置信息")


class ForwardRequest(BaseModel):
    """前向传播请求"""
    text: str = Field(..., description="输入文本", min_length=1, max_length=1000)
    capture_data: bool = Field(default=False, description="是否捕获中间数据")


class ForwardResponse(BaseModel):
    """前向传播响应"""
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="响应消息")
    logits_shape: List[int] = Field(..., description="输出logits的形状")
    sequence_length: int = Field(..., description="输入序列长度")
    captured_data: Optional[Dict[str, Any]] = Field(None, description="捕获的中间数据（如果请求）")