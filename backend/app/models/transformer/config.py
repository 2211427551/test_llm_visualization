"""
GPT-2模型配置

定义了Transformer模型的各种超参数，采用dataclass便于序列化和验证。
设计考虑了后续扩展性，支持稀疏注意力和MoE等高级功能。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPT2Config:
    """
    GPT-2风格Transformer模型的配置类
    
    这个配置类包含了模型所需的所有超参数，设计时参考了NanoGPT的配置方式，
    同时考虑了后续扩展稀疏注意力、MoE等高级功能的需求。
    
    Attributes:
        vocab_size: 词表大小，决定词嵌入层的维度
        context_size: 上下文窗口大小，即模型能处理的最大序列长度
        n_layer: Transformer层数，控制模型的深度
        n_head: 注意力头数，多头注意力的头数
        n_embed: 嵌入维度，隐藏层的维度大小
        dropout: Dropout概率，用于防止过拟合
        bias: 是否使用偏置项，在某些实现中可以省略偏置以提高效率
        
        # 前馈网络相关配置
        ffn_hidden_multiplier: 前馈网络隐藏层维度的倍数，通常设为4
        
        # 高级功能预留配置（为后续扩展做准备）
        use_sparse_attention: 是否使用稀疏注意力（预留）
        moe_num_experts: MoE专家数量（预留）
        moe_top_k: MoE路由时选择的专家数量（预留）
    """
    # 基础模型配置
    vocab_size: int = 50304  # 词表大小，通常是2的幂次，便于GPU优化
    context_size: int = 1024  # 上下文窗口大小
    n_layer: int = 12  # Transformer层数
    n_head: int = 12  # 注意力头数
    n_embed: int = 768  # 嵌入维度
    
    # 训练相关配置
    dropout: float = 0.1  # Dropout概率
    bias: bool = True  # 是否使用偏置项
    
    # 前馈网络配置
    ffn_hidden_multiplier: int = 4  # 前馈网络隐藏层维度倍数
    
    # 高级功能预留（为后续扩展做准备）
    use_sparse_attention: bool = False  # 稀疏注意力开关
    moe_num_experts: Optional[int] = None  # MoE专家数量
    moe_top_k: Optional[int] = None  # MoE路由top-k
    
    def __post_init__(self):
        """配置验证，确保参数的合理性"""
        # 验证嵌入维度能被注意力头数整除
        if self.n_embed % self.n_head != 0:
            raise ValueError(
                f"嵌入维度 {self.n_embed} 必须能被注意力头数 {self.n_head} 整除"
            )
        
        # 验证MoE配置的合理性
        if self.moe_num_experts is not None and self.moe_top_k is not None:
            if self.moe_top_k > self.moe_num_experts:
                raise ValueError(
                    f"MoE的top_k ({self.moe_top_k}) 不能大于专家数量 ({self.moe_num_experts})"
                )
    
    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.n_embed // self.n_head
    
    @property
    def ffn_hidden_size(self) -> int:
        """前馈网络隐藏层大小"""
        return self.n_embed * self.ffn_hidden_multiplier