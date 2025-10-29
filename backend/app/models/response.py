from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class InitialState(BaseModel):
    embeddings: List[List[float]] = Field(..., description="嵌入向量")
    positional_encodings: List[List[float]] = Field(..., description="位置编码")


class InitResponse(BaseModel):
    session_id: str = Field(..., description="会话ID")
    tokens: List[int] = Field(..., description="Token IDs")
    token_texts: List[str] = Field(..., description="Token文本")
    total_steps: int = Field(..., description="总步骤数")
    initial_state: InitialState = Field(..., description="初始状态")


class StepResponse(BaseModel):
    step: int = Field(..., description="当前步骤索引")
    step_type: str = Field(..., description="步骤类型")
    layer_index: int = Field(..., description="层索引")
    description: str = Field(..., description="步骤描述")
    input_data: Any = Field(..., description="输入数据")
    output_data: Any = Field(..., description="输出数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    attention_mask: Optional[List[List[float]]] = Field(default=None, description="注意力掩码矩阵")
    sparsity: Optional[float] = Field(default=None, description="稀疏度")
