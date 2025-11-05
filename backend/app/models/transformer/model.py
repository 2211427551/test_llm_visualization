"""
GPT-2风格Transformer模型主实现

整合了所有组件，构建完整的GPT-2风格Transformer模型：
- 嵌入层（词嵌入 + 位置编码）
- N层TransformerBlock
- 最终层归一化和输出投影

设计遵循GPT-2架构，支持训练和推理两种模式。
"""

from .embeddings import Embeddings
from .block import TransformerBlock

import torch
import torch.nn as nn


class GPT2Model(nn.Module):
    """
    GPT-2风格的Transformer模型
    
    这是一个仅解码器的Transformer模型，采用GPT-2的架构设计。
    主要特点：
    1. 可学习的位置编码
    2. Post-LN架构（层归一化在残差连接之后）
    3. 词嵌入与输出层权重共享
    4. 支持键值缓存加速推理
    
    模型架构：
    输入 -> 嵌入层 -> N×TransformerBlock -> 层归一化 -> 输出投影
    """
    
    def __init__(self, config):
        """
        初始化GPT-2模型
        
        Args:
            config: GPT2Config配置对象，包含所有模型超参数
        """
        super().__init__()
        self.config = config
        
        # 1. 嵌入层：词嵌入 + 位置编码
        # 将token序列转换为向量表示
        self.embeddings = Embeddings(config)
        
        # 2. Transformer块堆叠
        # 使用ModuleList便于逐层处理
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # 3. 最终层归一化
        # 在输出投影前进行归一化
        self.ln_f = nn.LayerNorm(config.n_embed)
        
        # 4. 输出投影层
        # 将隐藏状态映射回词表空间
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # 5. 权重绑定：词嵌入与输出层共享权重
        # 这是GPT-2的重要设计，减少参数并提高训练稳定性
        self.embeddings.tie_weights_with_output_layer(self.lm_head)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 特殊处理输出投影层的权重
        # 使用较小的标准差，提高训练稳定性
        self.lm_head.weight.data.normal_(mean=0.0, std=0.02)
    
    def _init_weights(self, module):
        """
        初始化模型权重
        
        使用GPT-2的权重初始化策略：
        1. 线性层：正态分布，标准差0.02
        2. 嵌入层：正态分布，标准差0.02
        3. 层归一化：权重为1，偏置为0
        
        Args:
            module: 要初始化的模块
        """
        if isinstance(module, nn.Linear):
            # 线性层权重初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层权重初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # 层归一化初始化
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        use_cache: bool = False,
        return_cache: bool = False,
        return_intermediate: bool = False,
        capture_container=None
    ) -> dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token序列，形状为 (batch_size, seq_len)
            use_cache: 是否使用键值缓存（用于推理加速）
            return_cache: 是否返回缓存
            return_intermediate: 是否返回中间张量（仅对稀疏注意力有效）
            capture_container: 可选的数据捕获容器
            
        Returns:
            dict包含：
            - logits: 输出logits，形状为 (batch_size, seq_len, vocab_size)
            - cache: 可选，键值缓存列表
            - intermediate: 可选，中间张量字典列表
            - trajectory: 可选，完整的前向传播轨迹（如果提供了capture_container）
        """
        batch_size, seq_len = input_ids.size()
        
        # 检查序列长度是否超过上下文窗口
        if seq_len > self.config.context_size:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过了上下文窗口大小 {self.config.context_size}"
            )
        
        # 1. 嵌入层处理
        # 输出: (batch_size, seq_len, n_embed)
        hidden_states = self.embeddings(input_ids, capture_container=capture_container)
        
        # 2. 通过Transformer块
        caches = [] if return_cache else None
        intermediate_tensors = [] if return_intermediate else None
        
        for i, block in enumerate(self.transformer_blocks):
            # 每个块都返回隐藏状态、可选的缓存和中间张量
            hidden_states, block_cache, block_intermediate = block(
                hidden_states, 
                use_cache=use_cache, 
                return_intermediate=return_intermediate,
                layer_idx=i,
                capture_container=capture_container
            )
            
            if return_cache and block_cache is not None:
                caches.append(block_cache)
            
            if return_intermediate and block_intermediate is not None:
                intermediate_tensors.append(block_intermediate)
        
        # 3. 最终层归一化
        hidden_states = self.ln_f(hidden_states)
        
        # 4. 输出投影
        # 输出: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(hidden_states)
        
        # 5. 捕获最终输出数据
        if capture_container is not None:
            capture_container.捕获最终输出(logits)
        
        result = {"logits": logits}
        if return_cache:
            result["cache"] = caches
        if return_intermediate:
            result["intermediate"] = intermediate_tensors
        if capture_container is not None:
            # 生成完整的前向传播轨迹
            result["trajectory"] = capture_container.生成完整轨迹(self, input_ids)
        
        return result
    
    def forward_with_capture(
        self,
        input_ids: torch.Tensor,
        capture_config=None,
        use_cache: bool = False,
        return_cache: bool = False,
        return_intermediate: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        带数据捕获的前向传播
        
        Args:
            input_ids: 输入token序列，形状为 (batch_size, seq_len)
            capture_config: 数据捕获配置
            use_cache: 是否使用键值缓存
            return_cache: 是否返回缓存
            return_intermediate: 是否返回中间张量
            
        Returns:
            包含标准输出和捕获数据的字典
        """
        from ...services.forward_capture import 数据捕获容器
        
        # 创建数据捕获容器
        capture_container = 数据捕获容器(self.config, capture_config)
        
        # 使用捕获上下文执行前向传播
        with capture_container.捕获上下文():
            result = self.forward(
                input_ids=input_ids,
                use_cache=use_cache,
                return_cache=return_cache,
                return_intermediate=return_intermediate,
                capture_container=capture_container
            )
        
        # 添加性能统计
        result["capture_stats"] = capture_container.获取性能统计()
        
        return result
    
    def clear_cache(self):
        """清除所有Transformer块的缓存"""
        for block in self.transformer_blocks:
            block.clear_cache()
    
    def get_num_parameters(self, only_trainable: bool = False) -> int:
        """
        获取模型参数数量
        
        Args:
            only_trainable: 是否只计算可训练参数
            
        Returns:
            参数数量
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())