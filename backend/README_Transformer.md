# GPT-2 Transformer骨干实现

本项目实现了GPT-2风格的仅解码器Transformer模型，采用模块化设计，参考NanoGPT架构，代码注释详细，便于理解和扩展。

## 📁 文件结构

```
backend/app/models/transformer/
├── __init__.py          # 模块导出
├── config.py            # GPT2Config配置类
├── attention.py         # 多头自注意力机制
├── mlp.py              # 前馈神经网络
├── block.py            # Transformer块
├── embeddings.py       # 词嵌入和位置编码
├── model.py            # 完整GPT-2模型
└── factory.py          # 工厂函数
```

## 🏗️ 架构特点

### 核心组件
- **词嵌入层**: 将token ID映射为向量表示
- **位置编码**: 可学习的位置嵌入（非正弦编码）
- **多头注意力**: 标准缩放点积注意力，支持因果掩码
- **前馈网络**: Linear -> GELU -> Linear -> Dropout
- **Transformer块**: 注意力 + MLP + 残差连接 + 层归一化

### 设计原则
- **Post-LN架构**: 层归一化在残差连接之后，训练更稳定
- **权重绑定**: 词嵌入与输出层共享权重，减少参数
- **缓存机制**: 支持键值缓存，加速推理
- **模块化设计**: 每个组件独立，便于测试和扩展

## 📋 使用示例

### 基础用法
```python
from app.models.transformer import GPT2Config, GPT2Model, create_gpt2_model

# 方法1: 使用配置类
config = GPT2Config(
    vocab_size=1000,
    context_size=512,
    n_layer=6,
    n_head=6,
    n_embed=384
)
model = GPT2Model(config)

# 方法2: 使用工厂函数
model = create_gpt2_model(
    vocab_size=1000,
    n_layer=6,
    n_head=6,
    n_embed=384
)

# 前向传播
input_ids = torch.randint(0, 1000, (2, 64))
result = model(input_ids)
logits = result["logits"]  # (2, 64, 1000)
```

### 预定义模型
```python
from app.models.transformer import create_gpt2_small, create_gpt2_medium

# 小型模型 (12层, 12头, 768维)
small_model = create_gpt2_small(vocab_size=1000)

# 中型模型 (24层, 16头, 1024维)
medium_model = create_gpt2_medium(vocab_size=1000)
```

### 带缓存的推理
```python
# 训练模式
result = model(input_ids, use_cache=False, return_cache=False)

# 推理模式（使用缓存）
result = model(input_ids, use_cache=True, return_cache=True)
logits = result["logits"]
cache = result["cache"]  # 用于下次推理
```

## 🔧 配置参数

### 基础参数
- `vocab_size`: 词表大小 (默认: 50304)
- `context_size`: 上下文窗口大小 (默认: 1024)
- `n_layer`: Transformer层数 (默认: 12)
- `n_head`: 注意力头数 (默认: 12)
- `n_embed`: 嵌入维度 (默认: 768)
- `dropout`: Dropout概率 (默认: 0.1)
- `bias`: 是否使用偏置项 (默认: True)

### 扩展参数（预留）
- `use_sparse_attention`: 稀疏注意力开关
- `moe_num_experts`: MoE专家数量
- `moe_top_k`: MoE路由top-k

## 🧪 测试验证

运行验证脚本：
```bash
cd backend
python verify_implementation.py
```

测试内容包括：
- ✅ 文件结构检查
- ✅ 代码组件验证
- ✅ 配置类测试
- ✅ 前向传播测试
- ✅ 权重绑定验证
- ✅ 工厂函数测试

## 📚 设计说明

### 为什么使用可学习位置编码？
1. **灵活性**: 相比固定的正弦位置编码，可学习的位置编码可以适应不同的数据分布
2. **性能**: 在许多NLP任务中，可学习的位置编码表现更好
3. **简洁性**: 实现简单，计算高效

### 为什么采用Post-LN架构？
1. **训练稳定性**: Post-LN在训练深层网络时更稳定
2. **性能表现**: GPT-2等现代语言模型普遍采用Post-LN架构
3. **梯度流**: 有助于梯度更好地传播到浅层

### 为什么权重绑定？
1. **参数效率**: 减少模型参数数量
2. **语义一致性**: 输入和输出空间保持一致，有助于训练稳定性
3. **训练稳定性**: GPT-2等模型验证了这种设计的有效性

## 🚀 扩展性

### 稀疏注意力
配置中已预留`use_sparse_attention`参数，可扩展实现：
- Local Attention
- Strided Attention
- Global Attention
- Longformer风格

### 混合专家（MoE）
配置中已预留MoE相关参数：
- `moe_num_experts`: 专家数量
- `moe_top_k`: 激活的专家数量
- 可扩展实现Router和Expert模块

### 其他扩展
- Rotary Position Embedding (RoPE)
- Flash Attention优化
- 量化支持
- 分布式训练

## 📊 模型规模对比

| 模型 | 层数 | 头数 | 嵌入维度 | 参数量 |
|------|------|------|----------|--------|
| Small | 12 | 12 | 768 | ~125M |
| Medium | 24 | 16 | 1024 | ~350M |
| Large | 36 | 20 | 1280 | ~760M |
| XL | 48 | 25 | 1600 | ~1.5B |

## 🎯 实现完成度

- ✅ **词嵌入**: 可学习的token嵌入
- ✅ **位置编码**: 可学习的位置嵌入
- ✅ **多头注意力**: 标准缩放点积注意力
- ✅ **前馈网络**: GELU激活的MLP
- ✅ **Transformer块**: 完整的解码器块
- ✅ **层归一化**: Post-LN架构
- ✅ **残差连接**: 标准残差连接
- ✅ **权重绑定**: 词嵌入与输出层共享
- ✅ **缓存机制**: 推理加速支持
- ✅ **配置系统**: 灵活的dataclass配置
- ✅ **工厂函数**: 便捷的模型创建
- ✅ **预定义模型**: 多种规模选择
- ✅ **单元测试**: 完整的功能验证
- ✅ **中文注释**: 详细的设计说明

## 📝 参考资料

- [NanoGPT](https://github.com/karpathy/nanoGPT) - 代码架构参考
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2论文

---

**实现完成时间**: 2024年
**代码风格**: 参考NanoGPT，模块化设计
**注释语言**: 中文，详细说明设计原理