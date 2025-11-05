"""
Mixture of Experts (MoE) 层实现

实现了一个完整的MoE层，包含：
- Gating网络：简单的线性层 + softmax
- Top-k路由：选择最相关的k个专家
- 多个并行专家：共享架构的前馈网络
- 中间数据捕获：路由得分、专家索引、专家输出等

设计遵循现代MoE架构的原则，确保可扩展性和训练稳定性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class MoEExpert(nn.Module):
    """
    MoE专家网络
    
    每个专家都是一个标准的前馈神经网络，与Transformer中的FFN架构相同。
    所有专家共享相同的架构，但参数独立。
    """
    
    def __init__(self, config):
        """
        初始化专家网络
        
        Args:
            config: GPT2Config配置对象
        """
        super().__init__()
        self.n_embed = config.n_embed
        self.ffn_hidden_size = config.ffn_hidden_size
        self.dropout = config.dropout if config.moe_dropout is None else config.moe_dropout
        self.bias = config.bias
        self.activation = config.moe_activation
        
        # 标准FFN架构：Linear -> Activation -> Linear -> Dropout
        self.c_fc = nn.Linear(self.n_embed, self.ffn_hidden_size, bias=self.bias)
        self.c_proj = nn.Linear(self.ffn_hidden_size, self.n_embed, bias=self.bias)
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embed)
            
        Returns:
            output: 输出张量，形状为 (batch_size, seq_len, n_embed)
        """
        x = self.c_fc(x)
        
        # 根据配置选择激活函数
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "swish":
            x = F.silu(x)  # SiLU就是Swish
        elif self.activation == "tanh":
            x = torch.tanh(x)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
        
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GatingNetwork(nn.Module):
    """
    Gating网络
    
    负责计算每个token对各个专家的权重分数。
    使用简单的线性层 + softmax实现。
    """
    
    def __init__(self, n_embed: int, num_experts: int, bias: bool = True):
        """
        初始化Gating网络
        
        Args:
            n_embed: 输入嵌入维度
            num_experts: 专家数量
            bias: 是否使用偏置项
        """
        super().__init__()
        self.gate = nn.Linear(n_embed, num_experts, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embed)
            
        Returns:
            gate_scores: 门控分数，形状为 (batch_size, seq_len, num_experts)
        """
        gate_logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        gate_scores = F.softmax(gate_logits, dim=-1)  # 沿专家维度softmax
        return gate_scores


class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) 层
    
    实现了完整的MoE机制，包含gating网络、Top-k路由和多个并行专家。
    
    核心功能：
    1. Gating网络计算每个token对各专家的权重
    2. Top-k路由选择最相关的k个专家
    3. 选中的专家处理对应的token
    4. 加权组合专家输出
    """
    
    def __init__(self, config, num_experts: int = 8, top_k: int = 2):
        """
        初始化MoE层
        
        Args:
            config: GPT2Config配置对象
            num_experts: 专家数量，默认为8
            top_k: 每个token选择的专家数量，默认为2
        """
        super().__init__()
        self.n_embed = config.n_embed
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = config.dropout
        
        # 验证参数合理性
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) 不能大于专家数量 ({num_experts})")
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            MoEExpert(config) for _ in range(num_experts)
        ])
        
        # 创建gating网络
        self.gating_network = GatingNetwork(
            self.n_embed, num_experts, bias=config.bias
        )
        
        # 负载均衡损失系数（用于训练时的负载均衡）
        self.load_balance_loss_coef = 0.01
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, n_embed)
            return_intermediate: 是否返回中间数据
            
        Returns:
            output: 输出张量，形状为 (batch_size, seq_len, n_embed)
            intermediate: 中间数据字典（如果return_intermediate为True）
        """
        batch_size, seq_len, n_embed = x.shape
        
        # 1. 计算gating分数
        gate_scores = self.gating_network(x)  # (batch_size, seq_len, num_experts)
        
        # 2. Top-k路由：选择最相关的k个专家
        top_k_gate_scores, top_k_indices = torch.topk(
            gate_scores, self.top_k, dim=-1, sorted=True
        )  # top_k_gate_scores: (batch_size, seq_len, top_k)
          # top_k_indices: (batch_size, seq_len, top_k)
        
        # 3. 归一化top-k分数
        top_k_gate_scores = top_k_gate_scores / (
            top_k_gate_scores.sum(dim=-1, keepdim=True) + 1e-8
        )  # 避免除零
        
        # 4. 初始化输出张量
        output = torch.zeros_like(x)  # (batch_size, seq_len, n_embed)
        
        # 5. 专家处理
        expert_outputs = []  # 存储每个专家的输出
        for expert_idx, expert in enumerate(self.experts):
            # 创建mask，标记哪些token被分配给当前专家
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (batch_size, seq_len)
            
            if expert_mask.any():
                # 获取需要当前专家处理的token
                expert_input = x[expert_mask]  # (selected_tokens, n_embed)
                
                # 获取对应的权重
                expert_weights = top_k_gate_scores[expert_mask]
                expert_weight_mask = (top_k_indices[expert_mask] == expert_idx)
                expert_weights = expert_weights[expert_weight_mask]  # (selected_tokens,)
                
                # 专家前向传播
                expert_output = expert(expert_input)  # (selected_tokens, n_embed)
                
                # 加权输出
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                
                # 将输出放回原位置
                output[expert_mask] += weighted_output
                
                # 保存专家输出（用于中间数据）
                if return_intermediate:
                    expert_outputs.append({
                        'expert_idx': expert_idx,
                        'mask': expert_mask,
                        'output': expert_output,
                        'weights': expert_weights
                    })
        
        # 6. 应用dropout
        output = F.dropout(output, p=self.dropout, training=self.training)
        
        # 7. 准备中间数据
        intermediate = None
        if return_intermediate:
            intermediate = {
                'gate_scores': gate_scores,  # 所有专家的门控分数
                'top_k_scores': top_k_gate_scores,  # Top-k分数
                'top_k_indices': top_k_indices,  # Top-k专家索引
                'expert_outputs': expert_outputs,  # 各专家输出
                'final_output': output,  # 最终加权输出
                'load_balance_loss': self.compute_load_balance_loss(gate_scores)
            }
        
        return output, intermediate
    
    def compute_load_balance_loss(self, gate_scores: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失
        
        鼓励所有专家被均匀使用，防止某些专家被过度使用而其他专家被忽略。
        
        Args:
            gate_scores: 门控分数，形状为 (batch_size, seq_len, num_experts)
            
        Returns:
            load_balance_loss: 负载均衡损失
        """
        # 计算每个专家的平均使用频率
        expert_usage = gate_scores.mean(dim=(0, 1))  # (num_experts,)
        
        # 理想情况下，每个专家的使用频率应该是1/num_experts
        ideal_usage = 1.0 / self.num_experts
        
        # 计算方差作为负载均衡损失
        load_balance_loss = torch.var(expert_usage - ideal_usage)
        
        return self.load_balance_loss_coef * load_balance_loss
    
    def get_expert_usage_stats(self, gate_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取专家使用统计信息
        
        Args:
            gate_scores: 门控分数
            
        Returns:
            stats: 包含各种统计信息的字典
        """
        # 每个专家的平均使用频率
        expert_usage = gate_scores.mean(dim=(0, 1))
        
        # 每个专家被选中的token数量
        _, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        expert_selections = torch.zeros(self.num_experts, device=gate_scores.device)
        for expert_idx in range(self.num_experts):
            expert_selections[expert_idx] = (top_k_indices == expert_idx).sum().float()
        
        return {
            'expert_usage': expert_usage,
            'expert_selections': expert_selections,
            'usage_std': torch.std(expert_usage),
            'selections_std': torch.std(expert_selections)
        }