from pydantic import BaseModel, Field
from app.utils.config import ModelConfig


class InitRequest(BaseModel):
    text: str = Field(..., description="输入文本")
    config: ModelConfig = Field(..., description="模型配置参数")


class StepRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    step: int = Field(..., ge=0, description="步骤索引")
