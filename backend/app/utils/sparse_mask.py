"""稀疏注意力掩码生成工具"""

import numpy as np
from typing import List, Optional


def generate_sliding_window_mask(n_tokens: int, window_size: int) -> np.ndarray:
    """
    生成滑动窗口稀疏掩码
    每个token只关注前后window_size个token
    
    Args:
        n_tokens: token数量
        window_size: 窗口大小
        
    Returns:
        mask: (n_tokens, n_tokens) 掩码矩阵，1表示允许关注，0表示屏蔽
    """
    mask = np.zeros((n_tokens, n_tokens), dtype=np.float32)
    for i in range(n_tokens):
        start = max(0, i - window_size)
        end = min(n_tokens, i + window_size + 1)
        mask[i, start:end] = 1
    return mask


def generate_global_local_mask(
    n_tokens: int, 
    window_size: int, 
    global_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    生成全局+局部稀疏掩码
    特殊token可以关注所有token，其他token只关注局部窗口
    
    Args:
        n_tokens: token数量
        window_size: 局部窗口大小
        global_indices: 全局token的索引列表
        
    Returns:
        mask: (n_tokens, n_tokens) 掩码矩阵
    """
    # 先生成滑动窗口掩码
    mask = generate_sliding_window_mask(n_tokens, window_size)
    
    if global_indices is None:
        # 默认第一个token为全局token (如CLS)
        global_indices = [0]
    
    # 全局token可以关注所有token
    for idx in global_indices:
        if 0 <= idx < n_tokens:
            mask[idx, :] = 1
            # 所有token也可以关注全局token
            mask[:, idx] = 1
    
    return mask


def generate_blocked_mask(n_tokens: int, block_size: int) -> np.ndarray:
    """
    生成分块稀疏掩码
    将序列分成块，token只关注同块和相邻块的token
    
    Args:
        n_tokens: token数量
        block_size: 块大小
        
    Returns:
        mask: (n_tokens, n_tokens) 掩码矩阵
    """
    mask = np.zeros((n_tokens, n_tokens), dtype=np.float32)
    n_blocks = (n_tokens + block_size - 1) // block_size
    
    for i in range(n_tokens):
        block_i = i // block_size
        
        # 关注同一块
        start_same = block_i * block_size
        end_same = min((block_i + 1) * block_size, n_tokens)
        mask[i, start_same:end_same] = 1
        
        # 关注前一块
        if block_i > 0:
            start_prev = (block_i - 1) * block_size
            end_prev = block_i * block_size
            mask[i, start_prev:end_prev] = 1
        
        # 关注后一块
        if block_i < n_blocks - 1:
            start_next = (block_i + 1) * block_size
            end_next = min((block_i + 2) * block_size, n_tokens)
            mask[i, start_next:end_next] = 1
    
    return mask


def generate_random_mask(n_tokens: int, random_ratio: float, seed: Optional[int] = None) -> np.ndarray:
    """
    生成随机稀疏掩码
    每个token随机选择一定比例的token进行关注
    
    Args:
        n_tokens: token数量
        random_ratio: 关注的比例 (0-1)
        seed: 随机种子
        
    Returns:
        mask: (n_tokens, n_tokens) 掩码矩阵
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机掩码
    mask = (np.random.rand(n_tokens, n_tokens) < random_ratio).astype(np.float32)
    
    # 确保每个token至少关注自己
    np.fill_diagonal(mask, 1)
    
    return mask


def generate_longformer_mask(
    n_tokens: int,
    window_size: int,
    global_indices: Optional[List[int]] = None,
    random_ratio: float = 0.05
) -> np.ndarray:
    """
    生成Longformer风格的稀疏掩码
    结合滑动窗口、全局关注和随机关注
    
    Args:
        n_tokens: token数量
        window_size: 滑动窗口大小
        global_indices: 全局token索引
        random_ratio: 随机关注比例
        
    Returns:
        mask: (n_tokens, n_tokens) 掩码矩阵
    """
    # 滑动窗口
    mask = generate_sliding_window_mask(n_tokens, window_size)
    
    # 全局token
    if global_indices is None:
        global_indices = [0]
    
    for idx in global_indices:
        if 0 <= idx < n_tokens:
            mask[idx, :] = 1
            mask[:, idx] = 1
    
    # 随机关注
    random_mask = (np.random.rand(n_tokens, n_tokens) < random_ratio).astype(np.float32)
    mask = np.maximum(mask, random_mask)
    
    return mask


def calculate_sparsity(mask: np.ndarray) -> float:
    """
    计算稀疏度（被屏蔽的比例）
    
    Args:
        mask: 掩码矩阵
        
    Returns:
        sparsity: 稀疏度 (0-1)
    """
    total_elements = mask.size
    masked_elements = np.sum(mask == 0)
    return float(masked_elements / total_elements)


def apply_mask_to_scores(scores: np.ndarray, mask: np.ndarray, mask_value: float = -1e9) -> np.ndarray:
    """
    将掩码应用到注意力分数
    被mask的位置设置为极小值（在softmax前）
    
    Args:
        scores: 注意力分数矩阵
        mask: 掩码矩阵 (1表示保留, 0表示屏蔽)
        mask_value: 被mask位置的值
        
    Returns:
        masked_scores: 应用掩码后的分数
    """
    masked_scores = scores.copy()
    masked_scores[mask == 0] = mask_value
    return masked_scores
