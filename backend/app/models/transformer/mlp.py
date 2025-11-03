"""
前馈神经网络（MLP）实现

实现了Transformer中的前馈神经网络部分，包含：
- 两个线性变换层
- GELU激活函数
- Dropout正则化

设计遵循GPT-2的标准配置，隐藏层维度通常是输入维度的4倍。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    前馈神经网络（MLP）
    
    实现了Transformer中的Position-wise Feed-Forward Network。
    标准配置是：Linear -> GELU -> Linear -> Dropout
    
    为什么这么设计：
    1. 维度扩展：第一个线性层将维度从n_embed扩展到4*n_embed，
       提供足够的表达能力来学习复杂的模式
    2. GELU激活：相比ReLU，GELU是平滑的激活函数，
       在Transformer中表现更好
    3. 维度还原：第二个线性层将维度还原到n_embed，
       保持残差连接的维度一致性
    4. Dropout：防止过拟合，提高泛化能力
    """
    
    def __init__(self, config):
        """
        初始化前馈网络
        
        Args:
            config: GPT2Config配置对象，包含n_embed、dropout等参数
        """
        super().__init__()
        self.n_embed = config.n_embed
        self.ffn_hidden_size = config.ffn_hidden_size
        self.dropout = config.dropout
        self.bias = config.bias
        
        # 第一个线性层：维度扩展 n_embed -> 4*n_embed
        # 使用大维度可以让模型学习更复杂的特征变换
        self.c_fc = nn.Linear(self.n_embed, self.ffn_hidden_size, bias=self.bias)
        
        # 第二个线性层：维度还原 4*n_embed -> n_embed
        # 将特征映射回原始维度，便于残差连接
        self.c_proj = nn.Linear(self.ffn_hidden_size, self.n_embed, bias=self.bias)
        
        # Dropout层，用于正则化
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embed)
            
        Returns:
            output: 输出张量，形状为 (batch_size, seq_len, n_embed)
        """
        # 1. 第一个线性变换：维度扩展
        # 输入: (batch_size, seq_len, n_embed)
        # 输出: (batch_size, seq_len, 4*n_embed)
        x = self.c_fc(x)
        
        # 2. GELU激活函数
        # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        # 相比ReLU，GELU是平滑的，允许梯度流过负值区域
        x = F.gelu(x)
        
        # 3. 第二个线性变换：维度还原
        # 输入: (batch_size, seq_len, 4*n_embed)
        # 输出: (batch_size, seq_len, n_embed)
        x = self.c_proj(x)
        
        # 4. Dropout正则化
        x = self.dropout(x)
        
        return x