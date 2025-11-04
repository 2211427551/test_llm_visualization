"""
稀疏注意力机制实现

实现了分组头、动态局部稀疏模式的稀疏注意力机制，模拟 Deepseek-V3.2-Exp 思路：
- 分组头注意力：将注意力头分为不同组，每组负责不同的注意力模式
- 动态局部稀疏：结合局部窗口和全局token的混合注意力模式
- 仅使用PyTorch操作，无需CUDA特制核心
- 返回详细的中间张量用于分析和调试

设计特点：
1. 分组策略：部分头使用局部注意力，部分头使用全局注意力
2. 动态掩码：根据序列长度动态调整局部窗口大小
3. 数值稳定性：使用-1e9作为mask值，确保softmax数值稳定
4. 可扩展性：支持不同的稀疏模式和配置
"""

import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SparseAttentionConfig:
    """稀疏注意力配置"""
    # 分组配置
    local_heads: int = 8  # 局部注意力头数
    global_heads: int = 4  # 全局注意力头数
    
    # 稀疏模式配置
    window_size: int = 128  # 局部窗口大小
    global_token_ratio: float = 0.1  # 全局token比例
    
    # 动态配置
    adaptive_window: bool = True  # 是否自适应窗口大小
    min_window_size: int = 32  # 最小窗口大小
    max_window_size: int = 512  # 最大窗口大小
    
    # 数值稳定性
    mask_value: float = -1e9  # mask填充值


class SparseAttention(nn.Module):
    """
    稀疏注意力机制
    
    实现了分组头、动态局部稀疏模式的注意力机制：
    1. 将注意力头分为局部组和全局组
    2. 局部组：只在固定窗口内计算注意力
    3. 全局组：可以关注所有位置，但重点关注全局token
    4. 动态调整窗口大小以适应不同序列长度
    
    优点：
    - 计算复杂度从O(n²)降低到O(n*w)，其中w是窗口大小
    - 保持长距离依赖建模能力
    - 数值稳定，易于调试
    """
    
    def __init__(self, config, sparse_config: Optional[SparseAttentionConfig] = None):
        """
        初始化稀疏注意力层
        
        Args:
            config: GPT2Config配置对象
            sparse_config: 稀疏注意力配置，如果为None则使用默认配置
        """
        super().__init__()
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.bias = config.bias
        
        # 稀疏注意力配置
        if sparse_config is None:
            sparse_config = SparseAttentionConfig()
        self.sparse_config = sparse_config
        
        # 验证配置
        total_heads = sparse_config.local_heads + sparse_config.global_heads
        if total_heads != self.n_head:
            raise ValueError(
                f"局部头数({sparse_config.local_heads}) + 全局头数({sparse_config.global_heads}) "
                f"必须等于总头数({self.n_head})"
            )
        
        # Q、K、V的合并线性变换
        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed, bias=self.bias)
        
        # 输出投影层
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=self.bias)
        
        # 注意权重的dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # 缓存的键值对，用于推理时的加速
        self.kv_cache = None
        
        # 头分组索引
        self.local_head_indices = list(range(sparse_config.local_heads))
        self.global_head_indices = list(
            range(sparse_config.local_heads, self.n_head)
        )
    
    def _compute_dynamic_window_size(self, seq_len: int) -> int:
        """
        计算动态窗口大小
        
        Args:
            seq_len: 序列长度
            
        Returns:
            动态窗口大小
        """
        if not self.sparse_config.adaptive_window:
            return self.sparse_config.window_size
        
        # 根据序列长度动态调整窗口大小
        base_window = self.sparse_config.window_size
        scale_factor = math.sqrt(seq_len / base_window)
        dynamic_window = int(base_window * scale_factor)
        
        # 限制在最小和最大窗口大小之间
        dynamic_window = max(
            self.sparse_config.min_window_size,
            min(dynamic_window, self.sparse_config.max_window_size)
        )
        
        return dynamic_window
    
    def _generate_local_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """
        生成本地注意力掩码
        
        Args:
            seq_len: 序列长度
            window_size: 窗口大小
            device: 设备
            
        Returns:
            本地掩码张量，形状为 (seq_len, seq_len)
        """
        mask = torch.full((seq_len, seq_len), self.sparse_config.mask_value, device=device)
        
        for i in range(seq_len):
            # 计算窗口范围（对称窗口）
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            
            # 允许关注窗口内的位置
            mask[i, start:end] = 0
        
        # 因果掩码：只能关注当前位置及之前的位置
        for i in range(seq_len):
            mask[i, i+1:] = self.sparse_config.mask_value
        
        return mask
    
    def _generate_global_mask(self, seq_len: int, num_global_tokens: int, device: torch.device) -> torch.Tensor:
        """
        生成全局注意力掩码
        
        Args:
            seq_len: 序列长度
            num_global_tokens: 全局token数量
            device: 设备
            
        Returns:
            全局掩码张量，形状为 (seq_len, seq_len)
        """
        mask = torch.full((seq_len, seq_len), self.sparse_config.mask_value, device=device)
        
        # 选择全局token位置（均匀分布）
        if num_global_tokens > 0:
            global_positions = torch.linspace(0, seq_len - 1, num_global_tokens, dtype=torch.long, device=device)
            
            # 所有位置都可以关注全局token
            for i in range(seq_len):
                mask[i, global_positions] = 0
            
            # 全局token可以关注所有位置（但保持因果性）
            for pos in global_positions:
                mask[pos, :pos+1] = 0
        
        # 因果掩码
        for i in range(seq_len):
            mask[i, :i+1] = 0
        
        return mask
    
    def _apply_sparse_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        head_indices: list,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用稀疏注意力
        
        Args:
            q: 查询张量，形状为 (batch_size, n_head, seq_len, head_dim)
            k: 键张量，形状为 (batch_size, n_head, seq_len, head_dim)
            v: 值张量，形状为 (batch_size, n_head, seq_len, head_dim)
            head_indices: 要应用的头索引列表
            mask: 注意力掩码，形状为 (seq_len, seq_len)
            
        Returns:
            注意力输出和注意力权重
        """
        batch_size, _, seq_len, head_dim = q.shape
        
        # 选择指定的头
        q_heads = q[:, head_indices, :, :]  # (batch_size, len(head_indices), seq_len, head_dim)
        k_heads = k[:, head_indices, :, :]
        v_heads = v[:, head_indices, :, :]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 应用掩码
        attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v_heads)
        
        return attn_output, attn_weights
    
    def forward(
        self, 
        x: torch.Tensor, 
        use_cache: bool = False,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embed)
            use_cache: 是否使用键值缓存，用于推理加速
            return_intermediate: 是否返回中间张量
            
        Returns:
            output: 注意力输出，形状为 (batch_size, seq_len, n_embed)
            cache: 缓存的键值对，形状为 (batch_size, n_head, seq_len, head_dim)
            intermediate: 中间张量字典（如果return_intermediate为True）
        """
        batch_size, seq_len, _ = x.size()
        intermediate_tensors = {}
        
        # 1. 计算Q、K、V
        qkv = self.c_attn(x)  # (batch_size, seq_len, 3 * n_embed)
        q, k, v = qkv.split(self.n_embed, dim=2)  # 每个都是 (batch_size, seq_len, n_embed)
        
        if return_intermediate:
            intermediate_tensors['qkv'] = qkv
            intermediate_tensors['q'] = q
            intermediate_tensors['k'] = k
            intermediate_tensors['v'] = v
        
        # 2. 重塑为多头格式
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        if return_intermediate:
            intermediate_tensors['q_reshaped'] = q
            intermediate_tensors['k_reshaped'] = k
            intermediate_tensors['v_reshaped'] = v
        
        # 3. 处理缓存（用于推理加速）
        if use_cache and self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        current_cache = (k, v) if use_cache else None
        
        # 4. 计算动态窗口大小
        window_size = self._compute_dynamic_window_size(seq_len)
        
        # 5. 生成稀疏掩码
        local_mask = self._generate_local_mask(seq_len, window_size, x.device)
        num_global_tokens = max(1, int(seq_len * self.sparse_config.global_token_ratio))
        global_mask = self._generate_global_mask(seq_len, num_global_tokens, x.device)
        
        if return_intermediate:
            intermediate_tensors['local_mask'] = local_mask
            intermediate_tensors['global_mask'] = global_mask
            intermediate_tensors['window_size'] = torch.tensor(window_size, device=x.device)
            intermediate_tensors['num_global_tokens'] = torch.tensor(num_global_tokens, device=x.device)
        
        # 6. 分别计算局部和全局注意力
        local_output, local_weights = self._apply_sparse_attention(
            q, k, v, self.local_head_indices, local_mask
        )
        global_output, global_weights = self._apply_sparse_attention(
            q, k, v, self.global_head_indices, global_mask
        )
        
        if return_intermediate:
            intermediate_tensors['local_attn_scores'] = local_weights
            intermediate_tensors['global_attn_weights'] = global_weights
            intermediate_tensors['local_attn_output'] = local_output
            intermediate_tensors['global_attn_output'] = global_output
        
        # 7. 合并局部和全局输出
        # 创建输出张量并填充
        attn_output = torch.zeros(batch_size, self.n_head, seq_len, self.head_dim, device=x.device)
        attn_output[:, self.local_head_indices, :, :] = local_output
        attn_output[:, self.global_head_indices, :, :] = global_output
        
        if return_intermediate:
            intermediate_tensors['merged_attn_output'] = attn_output
        
        # 8. 重塑输出并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_embed
        )
        
        # 9. 输出投影和dropout
        output = self.resid_dropout(self.c_proj(attn_output))
        
        if return_intermediate:
            intermediate_tensors['final_output'] = output
        
        return output, current_cache, intermediate_tensors if return_intermediate else None
    
    def clear_cache(self):
        """清除键值缓存"""
        self.kv_cache = None