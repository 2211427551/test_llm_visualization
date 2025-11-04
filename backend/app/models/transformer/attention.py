"""
多头自注意力机制实现

实现了GPT-2风格的多头自注意力机制，包含：
- 分割Q、K、V到多个头
- 缩放点积注意力
- 注意力掩码处理
- 输出投影

设计参考NanoGPT，注重代码的可读性和性能。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    
    实现了标准的缩放点积注意力，支持多头并行计算。
    设计上参考了GPT-2的实现方式，将Q、K、V的线性变换合并为一个矩阵乘法，
    然后分割成多个头，这样可以提高计算效率。
    
    为什么这么设计：
    1. 合并QKV线性变换：减少矩阵乘法次数，提高GPU利用率
    2. 使用Flash Attention：当可用时自动使用更高效的注意力实现
    3. 因果掩码：确保解码器只能看到当前位置之前的信息
    """
    
    def __init__(self, config):
        """
        初始化多头注意力层
        
        Args:
            config: GPT2Config配置对象，包含n_embed、n_head、dropout等参数
        """
        super().__init__()
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.bias = config.bias
        
        # Q、K、V的合并线性变换
        # 将三个权重矩阵合并为一个，提高计算效率
        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed, bias=self.bias)
        
        # 输出投影层
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=self.bias)
        
        # 注意权重的dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # 缓存的键值对，用于推理时的加速
        self.kv_cache = None
    
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
            output: 注意力输出，形状为 (batch_size, seq_len, n_embed)
            cache: 缓存的键值对，形状为 (batch_size, n_head, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. 计算Q、K、V
        # 合并的线性变换，然后分割成三个部分
        qkv = self.c_attn(x)  # (batch_size, seq_len, 3 * n_embed)
        q, k, v = qkv.split(self.n_embed, dim=2)  # 每个都是 (batch_size, seq_len, n_embed)
        
        # 2. 重塑为多头格式
        # (batch_size, seq_len, n_head, head_dim) -> (batch_size, n_head, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # 3. 处理缓存（用于推理加速）
        if use_cache and self.kv_cache is not None:
            # 将新的键值对与缓存的连接
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        # 更新缓存
        current_cache = (k, v) if use_cache else None
        
        # 4. 计算注意力
        # 使用PyTorch的scaled_dot_product_attention，它会自动选择最优实现
        # 包括Flash Attention等优化
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # 我们手动处理因果掩码
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True  # 启用因果掩码
        )
        
        # 5. 重塑输出并投影
        # (batch_size, n_head, seq_len, head_dim) -> (batch_size, seq_len, n_embed)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_embed
        )
        
        # 6. 输出投影和dropout
        output = self.resid_dropout(self.c_proj(attn_output))
        
        return output, current_cache
    
    def clear_cache(self):
        """清除键值缓存"""
        self.kv_cache = None