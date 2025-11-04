"""
稀疏注意力演示脚本

展示稀疏注意力模块的功能和特性：
- 配置稀疏注意力
- 对比标准注意力和稀疏注意力
- 展示中间张量
- 分析计算效率
"""

import sys
import os
import time

# 添加项目路径到Python路径
sys.path.insert(0, '/home/engine/project/backend')

import torch
from app.models.transformer import GPT2Config, GPT2Model
from app.models.transformer.sparse_attention import SparseAttentionConfig


def demo_sparse_attention():
    """演示稀疏注意力功能"""
    print("=" * 60)
    print("稀疏注意力模块演示")
    print("=" * 60)
    
    # 创建标准注意力模型配置
    standard_config = GPT2Config(
        vocab_size=1000,
        context_size=512,
        n_layer=2,
        n_embed=240,  # 240能被6整除
        n_head=6,
        use_sparse_attention=False
    )
    
    # 创建稀疏注意力模型配置
    sparse_config = GPT2Config(
        vocab_size=1000,
        context_size=512,
        n_layer=2,
        n_embed=240,  # 240能被6整除
        n_head=6,
        use_sparse_attention=True
    )
    
    print(f"配置信息:")
    print(f"  词表大小: {standard_config.vocab_size}")
    print(f"  上下文长度: {standard_config.context_size}")
    print(f"  层数: {standard_config.n_layer}")
    print(f"  嵌入维度: {standard_config.n_embed}")
    print(f"  注意力头数: {standard_config.n_head}")
    print(f"  头维度: {standard_config.head_dim}")
    
    # 创建模型
    print("\n创建模型...")
    standard_model = GPT2Model(standard_config)
    sparse_model = GPT2Model(sparse_config)
    
    print(f"标准模型参数量: {standard_model.get_num_parameters():,}")
    print(f"稀疏模型参数量: {sparse_model.get_num_parameters():,}")
    
    # 创建测试输入
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\n测试输入: batch_size={batch_size}, seq_len={seq_len}")
    
    # 前向传播对比
    print("\n前向传播对比:")
    
    # 标准注意力
    start_time = time.time()
    with torch.no_grad():
        standard_result = standard_model(input_ids)
    standard_time = time.time() - start_time
    
    # 稀疏注意力
    start_time = time.time()
    with torch.no_grad():
        sparse_result = sparse_model(input_ids)
    sparse_time = time.time() - start_time
    
    print(f"标准注意力时间: {standard_time:.4f}s")
    print(f"稀疏注意力时间: {sparse_time:.4f}s")
    print(f"加速比: {standard_time/sparse_time:.2f}x")
    
    # 输出统计
    print(f"\n输出统计:")
    std_logits = standard_result["logits"]
    sparse_logits = sparse_result["logits"]
    
    print(f"标准注意力输出: mean={std_logits.mean():.4f}, std={std_logits.std():.4f}")
    print(f"稀疏注意力输出: mean={sparse_logits.mean():.4f}, std={sparse_logits.std():.4f}")
    
    # 获取稀疏注意力的中间张量
    print(f"\n获取稀疏注意力中间张量...")
    with torch.no_grad():
        sparse_result_with_intermediate = sparse_model(
            input_ids, 
            return_intermediate=True
        )
    
    intermediate = sparse_result_with_intermediate["intermediate"]
    print(f"中间张量层数: {len(intermediate)}")
    
    # 分析第一层的稀疏特性
    layer_0_intermediate = intermediate[0]
    print(f"\n第一层稀疏特性分析:")
    print(f"  本地掩码形状: {layer_0_intermediate['local_mask'].shape}")
    print(f"  全局掩码形状: {layer_0_intermediate['global_mask'].shape}")
    print(f"  窗口大小: {layer_0_intermediate['window_size'].item()}")
    print(f"  全局token数: {layer_0_intermediate['num_global_tokens'].item()}")
    
    # 计算稀疏性
    local_mask = layer_0_intermediate['local_mask']
    total_elements = local_mask.numel()
    zero_elements = (local_mask == -1e9).sum().item()
    sparsity_ratio = zero_elements / total_elements
    
    print(f"  本地注意力稀疏性: {sparsity_ratio:.2%}")
    
    # 展示注意力权重分布
    local_weights = layer_0_intermediate['local_attn_scores']
    global_weights = layer_0_intermediate['global_attn_weights']
    
    print(f"  本地注意力权重形状: {local_weights.shape}")
    print(f"  全局注意力权重形状: {global_weights.shape}")
    
    # 不同序列长度的性能对比
    print(f"\n不同序列长度的性能对比:")
    seq_lengths = [64, 128, 256, 512]
    
    for seq_len in seq_lengths:
        test_input = torch.randint(0, 1000, (1, seq_len))
        
        # 标准注意力
        start_time = time.time()
        with torch.no_grad():
            standard_model(test_input)
        std_time = time.time() - start_time
        
        # 稀疏注意力
        start_time = time.time()
        with torch.no_grad():
            sparse_model(test_input)
        sparse_time = time.time() - start_time
        
        speedup = std_time / sparse_time
        print(f"  序列长度 {seq_len:3d}: 标准 {std_time:.4f}s, 稀疏 {sparse_time:.4f}s, 加速 {speedup:.2f}x")
    
    print(f"\n稀疏注意力配置详情:")
    sparse_attn = sparse_model.transformer_blocks[0].attn
    print(f"  局部头数: {len(sparse_attn.local_head_indices)}")
    print(f"  全局头数: {len(sparse_attn.global_head_indices)}")
    print(f"  窗口大小: {sparse_attn.sparse_config.window_size}")
    print(f"  自适应窗口: {sparse_attn.sparse_config.adaptive_window}")
    print(f"  全局token比例: {sparse_attn.sparse_config.global_token_ratio}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


def demo_mask_patterns():
    """演示不同的掩码模式"""
    print("\n" + "=" * 60)
    print("掩码模式演示")
    print("=" * 60)
    
    # 创建稀疏注意力
    gpt_config = GPT2Config(n_head=6, n_embed=240)
    sparse_config = SparseAttentionConfig(
        local_heads=4,
        global_heads=2,
        window_size=16,
        adaptive_window=False
    )
    
    from app.models.transformer.sparse_attention import SparseAttention
    sparse_attn = SparseAttention(gpt_config, sparse_config)
    
    seq_len = 20
    device = torch.device('cpu')
    
    # 生成本地和全局掩码
    local_mask = sparse_attn._generate_local_mask(seq_len, 16, device)
    global_mask = sparse_attn._generate_global_mask(seq_len, 3, device)
    
    print(f"序列长度: {seq_len}")
    print(f"窗口大小: 16")
    print(f"全局token数: 3")
    
    print(f"\n本地掩码 (0表示可访问, -1e9表示被遮蔽):")
    for i in range(min(10, seq_len)):
        row_str = ""
        for j in range(min(10, seq_len)):
            if local_mask[i, j] == 0:
                row_str += "  . "
            else:
                row_str += "  # "
        print(f"  {i:2d}: {row_str}")
    
    print(f"\n全局掩码 (0表示可访问, -1e9表示被遮蔽):")
    for i in range(min(10, seq_len)):
        row_str = ""
        for j in range(min(10, seq_len)):
            if global_mask[i, j] == 0:
                row_str += "  . "
            else:
                row_str += "  # "
        print(f"  {i:2d}: {row_str}")
    
    print("\n图例:")
    print("  . : 可访问的位置")
    print("  # : 被遮蔽的位置")


if __name__ == "__main__":
    try:
        demo_sparse_attention()
        demo_mask_patterns()
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()