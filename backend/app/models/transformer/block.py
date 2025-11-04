"""
Transformer块实现

实现了完整的TransformerBlock，包含：
- 多头自注意力机制
- 前馈神经网络
- 层归一化
- 残差连接

设计遵循GPT-2的Post-LN架构，将层归一化放在残差连接之后。
"""

from .attention import MultiHeadAttention
from .mlp import FeedForward
from .sparse_attention import SparseAttention, SparseAttentionConfig

import torch
import torch.nn as nn
from typing import Optional


class TransformerBlock(nn.Module):
    """
    Transformer块
    
    实现了标准的Transformer解码器块，包含两个主要的子层：
    1. 多头自注意力机制
    2. 前馈神经网络
    
    每个子层都采用残差连接和层归一化。
    
    为什么采用Post-LN架构：
    1. 训练稳定性：Post-LN（层归一化在残差连接之后）在训练深层网络时更稳定
    2. 性能表现：GPT-2等现代语言模型普遍采用Post-LN架构
    3. 梯度流：有助于梯度更好地传播到浅层
    """
    
    def __init__(self, config):
        """
        初始化Transformer块
        
        Args:
            config: GPT2Config配置对象，包含n_embed、n_head、dropout等参数
        """
        super().__init__()
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        
        # 1. 第一个层归一化：注意力前的归一化
        # 对输入进行归一化，提高训练稳定性
        self.ln_1 = nn.LayerNorm(self.n_embed)
        
        # 2. 注意力机制：根据配置选择标准注意力或稀疏注意力
        if config.use_sparse_attention:
            # 创建稀疏注意力配置
            sparse_config = SparseAttentionConfig(
                local_heads=config.n_head * 2 // 3,  # 2/3的头用于局部注意力
                global_heads=config.n_head // 3,     # 1/3的头用于全局注意力
                window_size=min(128, config.context_size // 4),
                adaptive_window=True,
            )
            self.attn = SparseAttention(config, sparse_config)
        else:
            self.attn = MultiHeadAttention(config)
        
        # 3. 第二个层归一化：MLP前的归一化
        self.ln_2 = nn.LayerNorm(self.n_embed)
        
        # 4. 前馈神经网络
        # 非线性变换模块
        self.mlp = FeedForward(config)
        
        # 注意力后的dropout
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        use_cache: bool = False,
        return_intermediate: bool = False
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]], Optional[dict]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embed)
            use_cache: 是否使用键值缓存，用于推理加速
            return_intermediate: 是否返回中间张量（仅对稀疏注意力有效）
            
        Returns:
            output: 输出张量，形状为 (batch_size, seq_len, n_embed)
            cache: 缓存的键值对，形状为 (batch_size, n_head, seq_len, head_dim)
            intermediate: 中间张量字典（如果return_intermediate为True）
        """
        # 保存原始输入用于残差连接
        original_x = x
        
        # 1. 第一个子层：注意力
        # Pre-LN: 先归一化，再注意力
        x = self.ln_1(x)
        
        # 根据注意力类型调用不同的前向传播
        if isinstance(self.attn, SparseAttention):
            attn_output, cache, intermediate = self.attn(x, use_cache=use_cache, return_intermediate=return_intermediate)
        else:
            attn_output, cache = self.attn(x, use_cache=use_cache)
            intermediate = None
        
        # 残差连接 + dropout
        x = original_x + self.resid_dropout(attn_output)
        
        # 保存当前状态用于第二个残差连接
        attn_x = x
        
        # 2. 第二个子层：前馈神经网络
        # Pre-LN: 先归一化，再MLP
        x = self.ln_2(x)
        mlp_output = self.mlp(x)
        
        # 残差连接 + dropout
        x = attn_x + self.resid_dropout(mlp_output)
        
        return x, cache, intermediate
    
    def clear_cache(self):
        """清除注意力层的缓存"""
        self.attn.clear_cache()