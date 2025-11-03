"""
模型工厂函数

提供便捷的模型创建函数，支持预定义配置和自定义配置。
设计为后续扩展预留接口，便于创建不同规模和功能的模型变体。
"""

from .config import GPT2Config
from .model import GPT2Model


def create_gpt2_model(
    vocab_size: int = 50304,
    context_size: int = 1024,
    n_layer: int = 12,
    n_head: int = 12,
    n_embed: int = 768,
    dropout: float = 0.1,
    bias: bool = True,
    **kwargs
) -> GPT2Model:
    """
    创建GPT-2模型的工厂函数
    
    提供便捷的模型创建接口，支持灵活的参数配置。
    设计考虑了易用性和扩展性，便于快速创建不同规模的模型。
    
    为什么使用工厂函数：
    1. 便捷性：用户无需手动创建配置对象
    2. 灵活性：支持部分参数覆盖
    3. 一致性：确保所有模型使用相同的创建流程
    4. 扩展性：便于后续添加预定义配置
    
    Args:
        vocab_size: 词表大小，默认50304（接近50257，但便于GPU优化）
        context_size: 上下文窗口大小，默认1024
        n_layer: Transformer层数，默认12
        n_head: 注意力头数，默认12
        n_embed: 嵌入维度，默认768
        dropout: Dropout概率，默认0.1
        bias: 是否使用偏置项，默认True
        **kwargs: 其他配置参数
        
    Returns:
        GPT2Model: 创建好的GPT-2模型实例
    """
    # 创建配置对象
    config = GPT2Config(
        vocab_size=vocab_size,
        context_size=context_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embed=n_embed,
        dropout=dropout,
        bias=bias,
        **kwargs
    )
    
    # 创建并返回模型
    return GPT2Model(config)


def create_gpt2_small(vocab_size: int = 50304, **kwargs) -> GPT2Model:
    """
    创建小型GPT-2模型（GPT-2 Small配置）
    
    配置：12层，12头，768维嵌入
    约1.25亿参数
    
    Args:
        vocab_size: 词表大小
        **kwargs: 其他配置参数
        
    Returns:
        GPT2Model: 小型GPT-2模型
    """
    return create_gpt2_model(
        n_layer=12,
        n_head=12,
        n_embed=768,
        vocab_size=vocab_size,
        **kwargs
    )


def create_gpt2_medium(vocab_size: int = 50304, **kwargs) -> GPT2Model:
    """
    创建中型GPT-2模型（GPT-2 Medium配置）
    
    配置：24层，16头，1024维嵌入
    约3.5亿参数
    
    Args:
        vocab_size: 词表大小
        **kwargs: 其他配置参数
        
    Returns:
        GPT2Model: 中型GPT-2模型
    """
    return create_gpt2_model(
        n_layer=24,
        n_head=16,
        n_embed=1024,
        vocab_size=vocab_size,
        **kwargs
    )


def create_gpt2_large(vocab_size: int = 50304, **kwargs) -> GPT2Model:
    """
    创建大型GPT-2模型（GPT-2 Large配置）
    
    配置：36层，20头，1280维嵌入
    约7.6亿参数
    
    Args:
        vocab_size: 词表大小
        **kwargs: 其他配置参数
        
    Returns:
        GPT2Model: 大型GPT-2模型
    """
    return create_gpt2_model(
        n_layer=36,
        n_head=20,
        n_embed=1280,
        vocab_size=vocab_size,
        **kwargs
    )


def create_gpt2_xl(vocab_size: int = 50304, **kwargs) -> GPT2Model:
    """
    创建超大型GPT-2模型（GPT-2 XL配置）
    
    配置：48层，25头，1600维嵌入
    约15亿参数
    
    Args:
        vocab_size: 词表大小
        **kwargs: 其他配置参数
        
    Returns:
        GPT2Model: 超大型GPT-2模型
    """
    return create_gpt2_model(
        n_layer=48,
        n_head=25,
        n_embed=1600,
        vocab_size=vocab_size,
        **kwargs
    )


# 预定义配置字典，便于快速选择
PREDEFINED_CONFIGS = {
    "small": {"n_layer": 12, "n_head": 12, "n_embed": 768},
    "medium": {"n_layer": 24, "n_head": 16, "n_embed": 1024},
    "large": {"n_layer": 36, "n_head": 20, "n_embed": 1280},
    "xl": {"n_layer": 48, "n_head": 25, "n_embed": 1600},
}


def create_gpt2_from_preset(preset: str, vocab_size: int = 50304, **kwargs) -> GPT2Model:
    """
    根据预设配置创建GPT-2模型
    
    Args:
        preset: 预设名称，可选 "small", "medium", "large", "xl"
        vocab_size: 词表大小
        **kwargs: 其他配置参数
        
    Returns:
        GPT2Model: 创建好的模型
        
    Raises:
        ValueError: 当预设名称无效时
    """
    if preset not in PREDEFINED_CONFIGS:
        raise ValueError(
            f"无效的预设名称 '{preset}'，可选：{list(PREDEFINED_CONFIGS.keys())}"
        )
    
    config = PREDEFINED_CONFIGS[preset]
    return create_gpt2_model(vocab_size=vocab_size, **config, **kwargs)