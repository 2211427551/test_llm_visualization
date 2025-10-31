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
    
    def get_sparsity_stats(self, attention_mask: List[List[float]]) -> dict:
        """计算稀疏度统计信息"""
        if not attention_mask:
            return {"sparsity": 0.0, "total_elements": 0, "zero_elements": 0}
        
        total_elements = sum(len(row) for row in attention_mask)
        zero_elements = sum(1 for row in attention_mask for val in row if val == 0.0)
        sparsity = zero_elements / total_elements if total_elements > 0 else 0.0
        
        return {
            "sparsity": sparsity,
            "total_elements": total_elements,
            "zero_elements": zero_elements,
            "nonzero_elements": total_elements - zero_elements
        }


class MoEConfig(BaseModel):
    """MoE (Mixture of Experts) 配置"""
    enabled: bool = Field(default=False, description="是否启用MoE")
    n_experts: int = Field(default=8, description="专家数量")
    top_k: int = Field(default=2, description="选择的top-k专家数量")
    gate_noise: float = Field(default=0.0, description="门控噪声")
    gate_dropout: float = Field(default=0.0, description="门控dropout率")
    
    def validate_config(self) -> bool:
        if self.enabled:
            if self.top_k > self.n_experts:
                raise ValueError("top_k不能大于n_experts")
            if self.n_experts <= 0:
                raise ValueError("n_experts必须大于0")
            if self.top_k <= 0:
                raise ValueError("top_k必须大于0")
        return True


class VisualizationConfig(BaseModel):
    """可视化配置"""
    render_mode: Literal["2d", "3d", "graph", "timeline"] = Field(
        default="2d", description="渲染模式"
    )
    animate_transitions: bool = Field(default=True, description="是否启用过渡动画")
    show_metadata: bool = Field(default=True, description="是否显示元数据")
    precision: int = Field(default=4, ge=1, le=8, description="浮点数精度")
    chunk_size: Optional[int] = Field(default=None, description="数据分块大小")
    enable_compression: bool = Field(default=True, description="是否启用压缩")


class ModelConfig(BaseModel):
    n_vocab: int = Field(default=50257, description="词汇表大小")
    n_embd: int = Field(default=768, description="嵌入维度")
    n_layer: int = Field(default=12, description="Transformer层数")
    n_head: int = Field(default=12, description="注意力头数")
    d_k: int = Field(default=64, description="每个注意力头的维度")
    max_seq_len: int = Field(default=512, description="最大序列长度")
    attention_type: Literal["standard", "sparse"] = Field(default="standard", description="注意力类型")
    sparse_config: Optional[SparseConfig] = Field(default=None, description="稀疏注意力配置")
    moe_config: Optional[MoEConfig] = Field(default=None, description="MoE配置")
    viz_config: Optional[VisualizationConfig] = Field(default=None, description="可视化配置")
    
    def validate_config(self) -> bool:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd必须能被n_head整除")
        if self.d_k != self.n_embd // self.n_head:
            raise ValueError("d_k应该等于n_embd // n_head")
        if self.attention_type == "sparse" and self.sparse_config is None:
            raise ValueError("稀疏注意力需要提供sparse_config")
        
        # 验证MoE配置
        if self.moe_config:
            self.moe_config.validate_config()
        
        return True


class SessionConfig:
    SESSION_TIMEOUT: int = 3600
    MAX_SESSIONS: int = 1000
