from pydantic import BaseModel, Field
from typing import Optional


class ModelConfig(BaseModel):
    n_vocab: int = Field(default=50257, description="词汇表大小")
    n_embd: int = Field(default=768, description="嵌入维度")
    n_layer: int = Field(default=12, description="Transformer层数")
    n_head: int = Field(default=12, description="注意力头数")
    d_k: int = Field(default=64, description="每个注意力头的维度")
    max_seq_len: int = Field(default=512, description="最大序列长度")
    
    def validate_config(self) -> bool:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd必须能被n_head整除")
        if self.d_k != self.n_embd // self.n_head:
            raise ValueError("d_k应该等于n_embd // n_head")
        return True


class SessionConfig:
    SESSION_TIMEOUT: int = 3600
    MAX_SESSIONS: int = 1000
