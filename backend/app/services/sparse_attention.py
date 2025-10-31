import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.utils.config import SparseConfig
from app.models.response import SparsityInfo


class SparseAttentionMask:
    """稀疏注意力掩码生成器"""
    
    def __init__(self, config: SparseConfig, seq_len: int):
        self.config = config
        self.seq_len = seq_len
        self.pattern = config.pattern
    
    def generate_mask(self) -> Tuple[np.ndarray, SparsityInfo]:
        """生成稀疏注意力掩码"""
        if self.pattern == "dense":
            mask = self._generate_dense_mask()
        elif self.pattern == "sliding_window":
            mask = self._generate_sliding_window_mask()
        elif self.pattern == "global_local":
            mask = self._generate_global_local_mask()
        elif self.pattern == "blocked":
            mask = self._generate_blocked_mask()
        elif self.pattern == "random":
            mask = self._generate_random_mask()
        elif self.pattern == "custom":
            mask = self._generate_custom_mask()
        else:
            mask = self._generate_dense_mask()
        
        # 计算稀疏度统计
        sparsity_info = self._compute_sparsity_info(mask)
        
        return mask, sparsity_info
    
    def _generate_dense_mask(self) -> np.ndarray:
        """生成稠密掩码（全1）"""
        return np.ones((self.seq_len, self.seq_len), dtype=np.float32)
    
    def _generate_sliding_window_mask(self) -> np.ndarray:
        """生成滑动窗口掩码"""
        window_size = self.config.window_size or 3
        mask = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)
        
        for i in range(self.seq_len):
            start = max(0, i - window_size)
            end = min(self.seq_len, i + window_size + 1)
            mask[i, start:end] = 1.0
        
        return mask
    
    def _generate_global_local_mask(self) -> np.ndarray:
        """生成全局+局部掩码"""
        window_size = self.config.window_size or 3
        global_tokens = self.config.global_tokens or list(range(min(4, self.seq_len)))
        
        mask = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)
        
        # 滑动窗口部分
        for i in range(self.seq_len):
            start = max(0, i - window_size)
            end = min(self.seq_len, i + window_size + 1)
            mask[i, start:end] = 1.0
        
        # 全局token部分
        for global_token in global_tokens:
            if global_token < self.seq_len:
                mask[:, global_token] = 1.0
                mask[global_token, :] = 1.0
        
        return mask
    
    def _generate_blocked_mask(self) -> np.ndarray:
        """生成分块掩码"""
        block_size = self.config.block_size or 4
        mask = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)
        
        for i in range(0, self.seq_len, block_size):
            for j in range(0, self.seq_len, block_size):
                # 在每个block内部允许注意力
                end_i = min(i + block_size, self.seq_len)
                end_j = min(j + block_size, self.seq_len)
                mask[i:end_i, j:end_j] = 1.0
        
        return mask
    
    def _generate_random_mask(self) -> np.ndarray:
        """生成随机稀疏掩码"""
        random_ratio = self.config.random_ratio or 0.1
        
        # 基础掩码（包含对角线）
        mask = np.eye(self.seq_len, dtype=np.float32)
        
        # 添加随机连接
        total_elements = self.seq_len * self.seq_len
        random_elements = int(total_elements * random_ratio)
        
        # 随机选择位置
        flat_indices = np.random.choice(
            total_elements, 
            size=min(random_elements, total_elements - self.seq_len), 
            replace=False
        )
        
        for idx in flat_indices:
            i = idx // self.seq_len
            j = idx % self.seq_len
            mask[i, j] = 1.0
        
        return mask
    
    def _generate_custom_mask(self) -> np.ndarray:
        """生成自定义掩码"""
        if self.config.custom_mask:
            return np.array(self.config.custom_mask, dtype=np.float32)
        else:
            # 如果没有提供自定义掩码，回退到稠密掩码
            return self._generate_dense_mask()
    
    def _compute_sparsity_info(self, mask: np.ndarray) -> SparsityInfo:
        """计算稀疏度信息"""
        total_elements = mask.size
        zero_elements = np.sum(mask == 0.0)
        nonzero_elements = total_elements - zero_elements
        sparsity_ratio = zero_elements / total_elements if total_elements > 0 else 0.0
        
        return SparsityInfo(
            pattern=self.pattern,
            mask_matrix=mask.tolist(),
            sparsity_ratio=float(sparsity_ratio),
            total_elements=int(total_elements),
            zero_elements=int(zero_elements),
            nonzero_elements=int(nonzero_elements)
        )


class SparseAttentionSimulator:
    """稀疏注意力模拟器"""
    
    def __init__(self, n_embd: int, n_head: int, config: SparseConfig):
        self.n_embd = n_embd
        self.n_head = n_head
        self.config = config
    
    def apply_sparse_mask(self, attention_weights: np.ndarray, 
                         mask: np.ndarray) -> np.ndarray:
        """应用稀疏掩码到注意力权重"""
        # 确保形状匹配
        if attention_weights.shape != mask.shape:
            raise ValueError(f"注意力权重形状 {attention_weights.shape} 与掩码形状 {mask.shape} 不匹配")
        
        # 应用掩码
        masked_weights = attention_weights * mask
        
        # 重新归一化
        row_sums = np.sum(masked_weights, axis=-1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # 避免除零
        normalized_weights = masked_weights / row_sums
        
        return normalized_weights
    
    def simulate_sparse_attention(self, x: np.ndarray, seq_len: int):
        """模拟稀疏注意力计算"""
        steps = []
        
        # 生成稀疏掩码
        mask_generator = SparseAttentionMask(self.config, seq_len)
        attention_mask, sparsity_info = mask_generator.generate_mask()
        steps.append(("sparse_mask", attention_mask, {"pattern": self.config.pattern}))
        
        # 模拟注意力权重计算
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        n_head = self.n_head
        d_k = self.n_embd // n_head
        
        # 生成随机注意力权重（模拟）
        attention_weights = np.random.randn(batch_size, n_head, seq_len, seq_len)
        attention_weights = self._softmax(attention_weights, axis=-1)
        steps.append(("attention_weights", attention_weights.mean(axis=1), {"shape": attention_weights.shape}))
        
        # 应用稀疏掩码
        masked_weights = np.zeros_like(attention_weights.mean(axis=1))
        for head in range(n_head):
            masked_weights += self.apply_sparse_mask(attention_weights[head], attention_mask)
        masked_weights /= n_head
        
        steps.append(("masked_attention", masked_weights, {"sparsity_applied": True}))
        
        # 计算输出
        V = np.random.randn(seq_len, self.n_embd) * 0.02  # 模拟Value矩阵
        output = np.dot(masked_weights, V)
        steps.append(("sparse_attention_output", output, {"shape": output.shape}))
        
        return output, sparsity_info, steps
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax函数"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)