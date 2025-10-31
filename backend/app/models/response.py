from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class InitialState(BaseModel):
    embeddings: List[List[float]] = Field(..., description="嵌入向量")
    positional_encodings: List[List[float]] = Field(..., description="位置编码")


class ExpertInfo(BaseModel):
    expert_id: int = Field(..., description="专家ID")
    gate_probability: float = Field(..., description="门控概率")
    output: List[List[float]] = Field(..., description="专家输出")


class MoERouting(BaseModel):
    top_k_experts: List[ExpertInfo] = Field(..., description="Top-K专家信息")
    gate_logits: List[float] = Field(..., description="门控logits")
    combined_output: List[List[float]] = Field(..., description="组合输出")


class SparsityInfo(BaseModel):
    pattern: str = Field(..., description="稀疏模式")
    mask_matrix: List[List[float]] = Field(..., description="掩码矩阵")
    sparsity_ratio: float = Field(..., description="稀疏度比例")
    total_elements: int = Field(..., description="总元素数")
    zero_elements: int = Field(..., description="零元素数")
    nonzero_elements: int = Field(..., description="非零元素数")


class InitResponse(BaseModel):
    session_id: str = Field(..., description="会话ID")
    tokens: List[int] = Field(..., description="Token IDs")
    token_texts: List[str] = Field(..., description="Token文本")
    total_steps: int = Field(..., description="总步骤数")
    initial_state: InitialState = Field(..., description="初始状态")
    sparse_info: Optional[SparsityInfo] = Field(default=None, description="稀疏信息")
    moe_info: Optional[Dict[str, Any]] = Field(default=None, description="MoE信息")


class StepResponse(BaseModel):
    step: int = Field(..., description="当前步骤索引")
    step_type: str = Field(..., description="步骤类型")
    layer_index: int = Field(..., description="层索引")
    description: str = Field(..., description="步骤描述")
    input_data: Any = Field(..., description="输入数据")
    output_data: Any = Field(..., description="输出数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    attention_mask: Optional[List[List[float]]] = Field(default=None, description="注意力掩码矩阵")
    sparsity_info: Optional[SparsityInfo] = Field(default=None, description="稀疏信息")
    moe_routing: Optional[MoERouting] = Field(default=None, description="MoE路由信息")


class TraceResponse(BaseModel):
    session_id: str = Field(..., description="会话ID")
    input_text: str = Field(..., description="输入文本")
    tokens: List[int] = Field(..., description="Token IDs")
    token_texts: List[str] = Field(..., description="Token文本")
    total_steps: int = Field(..., description="总步骤数")
    computation_trace: List[StepResponse] = Field(..., description="完整计算轨迹")
    final_output: List[List[float]] = Field(..., description="最终输出")
    execution_time_ms: float = Field(..., description="执行时间(毫秒)")


class ChunkedResponse(BaseModel):
    chunk_id: int = Field(..., description="分块ID")
    total_chunks: int = Field(..., description="总分块数")
    data: Any = Field(..., description="分块数据")
    has_more: bool = Field(..., description="是否还有更多数据")


class StreamMessage(BaseModel):
    type: str = Field(..., description="消息类型: step, error, complete")
    data: Any = Field(..., description="消息数据")
    timestamp: Optional[float] = Field(default=None, description="时间戳")
