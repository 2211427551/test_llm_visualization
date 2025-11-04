"""
Transformer模块

实现GPT-2风格的仅解码器Transformer模型，包含：
- 词嵌入层
- 可学习位置编码
- 多层TransformerBlock
- 标准多头自注意力机制
- 前馈神经网络

设计参考NanoGPT，采用模块化设计，便于后续扩展稀疏注意力、MoE等功能。
"""

from .config import GPT2Config
from .model import GPT2Model
from .factory import create_gpt2_model

__all__ = ["GPT2Config", "GPT2Model", "create_gpt2_model"]