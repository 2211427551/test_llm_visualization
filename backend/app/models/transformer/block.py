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

import torch
import torch.nn as nn


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
        
        # 2. 多头自注意力机制
        # 核心的注意力计算模块
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
        use_cache: bool = False
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embed)
            use_cache: 是否使用键值缓存，用于推理加速
            
        Returns:
            output: 输出张量，形状为 (batch_size, seq_len, n_embed)
            cache: 缓存的键值对，形状为 (batch_size, n_head, seq_len, head_dim)
        """
        # 保存原始输入用于残差连接
        original_x = x
        
        # 1. 第一个子层：多头自注意力
        # Pre-LN: 先归一化，再注意力
        x = self.ln_1(x)
        attn_output, cache = self.attn(x, use_cache=use_cache)
        
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
        
        return x, cache
    
    def clear_cache(self):
        """清除注意力层的缓存"""
        self.attn.clear_cache()