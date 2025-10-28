from pydantic import BaseModel, Field
from typing import Optional, Literal, List


class SparseConfig(BaseModel):
    """稀疏注意力配置"""
    pattern: Literal["dense", "sliding_window", "global_local", "blocked", "random", "custom"] = Field(
        default="dense",
        description="稀疏模式"
    )
    window_size: Optional[int] = Field(default=3, description="滑动窗口大小")
    block_size: Optional[int] = Field(default=4, description="分块大小")
    global_tokens: Optional[List[int]] = Field(default=None, description="全局token索引")
    random_ratio: Optional[float] = Field(default=0.1, description="随机稀疏比例")
    custom_mask: Optional[List[List[int]]] = Field(default=None, description="自定义掩码矩阵")


class ModelConfig(BaseModel):
    n_vocab: int = Field(default=50257, description="词汇表大小")
    n_embd: int = Field(default=768, description="嵌入维度")
    n_layer: int = Field(default=12, description="Transformer层数")
    n_head: int = Field(default=12, description="注意力头数")
    d_k: int = Field(default=64, description="每个注意力头的维度")
    max_seq_len: int = Field(default=512, description="最大序列长度")
    attention_type: Literal["standard", "sparse"] = Field(default="standard", description="注意力类型")
    sparse_config: Optional[SparseConfig] = Field(default=None, description="稀疏注意力配置")
    
    def validate_config(self) -> bool:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd必须能被n_head整除")
        if self.d_k != self.n_embd // self.n_head:
            raise ValueError("d_k应该等于n_embd // n_head")
        if self.attention_type == "sparse" and self.sparse_config is None:
            raise ValueError("稀疏注意力需要提供sparse_config")
        return True


class SessionConfig:
    SESSION_TIMEOUT: int = 3600
    MAX_SESSIONS: int = 1000
