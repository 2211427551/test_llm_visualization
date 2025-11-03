"""
Models package

包含各种深度学习模型实现，主要包括：
- transformer: GPT-2风格的Transformer模型
"""

from .transformer import GPT2Config, GPT2Model, create_gpt2_model

__all__ = ["GPT2Config", "GPT2Model", "create_gpt2_model"]