"""
嵌入层实现

实现了GPT-2风格的嵌入层，包含：
- 词嵌入
- 可学习位置编码
- 嵌入dropout

设计参考GPT-2，使用可学习的位置编码而非正弦位置编码。
"""

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """
    嵌入层模块
    
    实现了词嵌入和位置编码的组合，这是Transformer模型的第一层。
    
    为什么使用可学习位置编码：
    1. 灵活性：相比固定的正弦位置编码，可学习的位置编码可以适应不同的数据分布
    2. 性能：在许多NLP任务中，可学习的位置编码表现更好
    3. 简洁性：实现简单，计算高效
    
    为什么词嵌入和最终输出层共享权重：
    1. 参数效率：减少模型参数数量
    2. 语义一致性：输入和输出空间保持一致，有助于训练稳定性
    """
    
    def __init__(self, config):
        """
        初始化嵌入层
        
        Args:
            config: GPT2Config配置对象，包含vocab_size、n_embed、context_size等参数
        """
        super().__init__()
        self.vocab_size = config.vocab_size
        self.n_embed = config.n_embed
        self.context_size = config.context_size
        self.dropout = config.dropout
        
        # 词嵌入层：将词汇ID映射到向量空间
        # 输入: (batch_size, seq_len) -> 输出: (batch_size, seq_len, n_embed)
        self.wte = nn.Embedding(self.vocab_size, self.n_embed)
        
        # 位置编码：为每个位置提供可学习的嵌入向量
        # 输入: (batch_size, seq_len) -> 输出: (batch_size, seq_len, n_embed)
        self.wpe = nn.Embedding(self.context_size, self.n_embed)
        
        # Dropout层，用于嵌入的正则化
        self.drop = nn.Dropout(self.dropout)
    
    def forward(self, input_ids: torch.Tensor, capture_container=None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID序列，形状为 (batch_size, seq_len)
            capture_container: 可选的数据捕获容器
            
        Returns:
            embeddings: 融合了词嵌入和位置编码的嵌入向量，
                      形状为 (batch_size, seq_len, n_embed)
        """
        batch_size, seq_len = input_ids.size()
        
        # 1. 获取词嵌入
        # 查找每个token对应的嵌入向量
        token_embeddings = self.wte(input_ids)  # (batch_size, seq_len, n_embed)
        
        # 2. 获取位置编码
        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_ids.device)
        position_embeddings = self.wpe(positions)  # (seq_len, n_embed)
        
        # 扩展位置编码到batch维度
        position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. 融合词嵌入和位置编码
        # 直接相加，这是Transformer的标准做法
        embeddings = token_embeddings + position_embeddings
        
        # 4. 应用dropout
        embeddings = self.drop(embeddings)
        
        # 5. 数据捕获（如果提供了捕获容器）
        if capture_container is not None:
            capture_container.捕获嵌入数据(
                input_ids=input_ids,
                token_embeddings=token_embeddings,
                position_embeddings=position_embeddings,
                final_embeddings=embeddings
            )
        
        return embeddings
    
    def tie_weights_with_output_layer(self, output_layer: nn.Linear):
        """
        将词嵌入权重与输出层权重绑定
        
        这是GPT-2的一个重要设计，可以：
        1. 减少参数数量
        2. 提高训练稳定性
        3. 保持输入输出空间的一致性
        
        Args:
            output_layer: 输出层的线性变换层
        """
        self.wte.weight = output_layer.weight