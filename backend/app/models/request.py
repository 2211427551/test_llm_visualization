from pydantic import BaseModel, Field
from app.utils.config import ModelConfig


class InitRequest(BaseModel):
    text: str = Field(..., description="输入文本")
    config: ModelConfig = Field(..., description="模型配置参数")


class StepRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    step: int = Field(..., ge=0, description="步骤索引")


class TraceRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    precision: Optional[int] = Field(default=4, ge=1, le=8, description="浮点数精度")


class ChunkRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    chunk_size: int = Field(default=100, ge=1, le=1000, description="分块大小")
    chunk_id: int = Field(default=0, ge=0, description="分块ID")


class WebSocketMessage(BaseModel):
    type: str = Field(..., description="消息类型: subscribe, unsubscribe, step, trace")
    session_id: str = Field(..., description="会话ID")
    data: Optional[Dict[str, Any]] = Field(default=None, description="消息数据")
