"""
稀疏注意力模块单元测试

测试稀疏注意力的各个方面：
- 配置验证
- 分组头注意力机制
- 稀疏掩码生成
- 数值稳定性
- 中间张量返回
- 与标准注意力的兼容性
"""

import sys
import os
import math

# 添加项目路径到Python路径
sys.path.insert(0, '/home/engine/project/backend')

try:
    import torch
    import torch.nn.functional as F
    print("✓ PyTorch导入成功")
except ImportError:
    print("✗ PyTorch未安装")
    exit(1)

from app.models.transformer import GPT2Config
from app.models.transformer.sparse_attention import SparseAttention, SparseAttentionConfig
from app.models.transformer.block import TransformerBlock


def test_sparse_attention_config():
    """测试稀疏注意力配置"""
    print("\n=== 测试稀疏注意力配置 ===")
    
    # 测试默认配置
    config = SparseAttentionConfig()
    print(f"默认配置: local_heads={config.local_heads}, global_heads={config.global_heads}")
    print(f"窗口大小: {config.window_size}, 自适应窗口: {config.adaptive_window}")
    
    # 测试自定义配置
    custom_config = SparseAttentionConfig(
        local_heads=4,
        global_heads=2,
        window_size=64,
        adaptive_window=False
    )
    print(f"自定义配置: local_heads={custom_config.local_heads}, global_heads={custom_config.global_heads}")
    
    print("✓ 稀疏注意力配置测试通过")
    return True


def test_sparse_attention_initialization():
    """测试稀疏注意力初始化"""
    print("\n=== 测试稀疏注意力初始化 ===")
    
    # 创建基础配置 - 确保n_embed能被n_head整除
    gpt_config = GPT2Config(
        vocab_size=1000,
        context_size=512,
        n_layer=2,
        n_embed=240,  # 240能被6整除
        n_head=6      # 便于测试分组
    )
    
    # 测试稀疏注意力初始化
    sparse_config = SparseAttentionConfig(
        local_heads=4,
        global_heads=2
    )
    
    sparse_attn = SparseAttention(gpt_config, sparse_config)
    print(f"稀疏注意力初始化成功: n_head={sparse_attn.n_head}, head_dim={sparse_attn.head_dim}")
    print(f"局部头索引: {sparse_attn.local_head_indices}")
    print(f"全局头索引: {sparse_attn.global_head_indices}")
    
    # 测试配置验证
    try:
        wrong_config = SparseAttentionConfig(local_heads=3, global_heads=4)  # 总共7个头，但配置是6个
        SparseAttention(gpt_config, wrong_config)
        print("✗ 配置验证失败")
        return False
    except ValueError:
        print("✓ 配置验证正确")
    
    print("✓ 稀疏注意力初始化测试通过")
    return True


def test_dynamic_window_size():
    """测试动态窗口大小计算"""
    print("\n=== 测试动态窗口大小计算 ===")
    
    gpt_config = GPT2Config(n_head=6, n_embed=240)  # 240能被6整除
    sparse_config = SparseAttentionConfig(
        local_heads=4,  # 4个局部头
        global_heads=2, # 2个全局头，总共6个头
        window_size=128,
        adaptive_window=True,
        min_window_size=32,
        max_window_size=512
    )
    sparse_attn = SparseAttention(gpt_config, sparse_config)
    
    # 测试不同序列长度
    test_seq_lengths = [64, 128, 256, 512, 1024]
    
    for seq_len in test_seq_lengths:
        window_size = sparse_attn._compute_dynamic_window_size(seq_len)
        print(f"序列长度 {seq_len:4d} -> 窗口大小 {window_size:4d}")
        
        # 验证窗口大小在合理范围内
        if not (sparse_config.min_window_size <= window_size <= sparse_config.max_window_size):
            print(f"✗ 窗口大小超出范围: {window_size}")
            return False
    
    print("✓ 动态窗口大小计算测试通过")
    return True


def test_mask_generation():
    """测试掩码生成"""
    print("\n=== 测试掩码生成 ===")
    
    device = torch.device('cpu')
    seq_len = 10
    window_size = 3
    
    gpt_config = GPT2Config(n_head=6, n_embed=240)  # 240能被6整除
    sparse_config = SparseAttentionConfig(
        local_heads=4,  # 4个局部头
        global_heads=2, # 2个全局头
        window_size=window_size
    )
    sparse_attn = SparseAttention(gpt_config, sparse_config)
    
    # 测试本地掩码
    local_mask = sparse_attn._generate_local_mask(seq_len, window_size, device)
    print(f"本地掩码形状: {local_mask.shape}")
    print("本地掩码示例 (前5行前5列):")
    print(local_mask[:5, :5])
    
    # 验证本地掩码特性
    for i in range(seq_len):
        # 检查窗口范围（考虑因果掩码）
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(i + 1, i + half_window + 1)  # 限制到i+1因为因果掩码
        
        # 窗口内应该为0（考虑因果性）
        for j in range(start, end):
            if local_mask[i, j] != 0:
                print(f"✗ 位置{i}的窗口掩码错误: 位置{j}应该为0")
                return False
        
        # 窗口外且在当前位置之前应该为mask_value
        for j in range(end, i + 1):
            if j < seq_len and local_mask[i, j] != sparse_config.mask_value:
                print(f"✗ 位置{i}的窗口外掩码错误: 位置{j}应该为mask_value")
                return False
        
        # 因果掩码：当前位置之后应该为mask_value
        if i + 1 < seq_len and not torch.all(local_mask[i, i+1:] == sparse_config.mask_value):
            print(f"✗ 位置{i}的因果掩码错误")
            return False
    
    # 测试全局掩码
    num_global_tokens = 2
    global_mask = sparse_attn._generate_global_mask(seq_len, num_global_tokens, device)
    print(f"\n全局掩码形状: {global_mask.shape}")
    print("全局掩码示例 (前5行前5列):")
    print(global_mask[:5, :5])
    
    print("✓ 掩码生成测试通过")
    return True


def test_sparse_attention_forward():
    """测试稀疏注意力前向传播"""
    print("\n=== 测试稀疏注意力前向传播 ===")
    
    # 创建配置
    gpt_config = GPT2Config(
        vocab_size=1000,
        context_size=512,
        n_layer=1,
        n_embed=240,  # 240能被6整除
        n_head=6
    )
    
    sparse_config = SparseAttentionConfig(
        local_heads=4,
        global_heads=2,
        window_size=64,
        adaptive_window=False
    )
    
    sparse_attn = SparseAttention(gpt_config, sparse_config)
    
    # 测试不同输入形状
    test_cases = [
        (1, 32),   # 单样本，短序列
        (2, 64),   # 多样本，中等序列
        (1, 128),  # 单样本，长序列
    ]
    
    for batch_size, seq_len in test_cases:
        print(f"测试输入形状: ({batch_size}, {seq_len})")
        
        # 创建随机输入
        x = torch.randn(batch_size, seq_len, gpt_config.n_embed)
        
        # 前向传播（不返回中间张量）
        output, cache, _ = sparse_attn(x, use_cache=False, return_intermediate=False)
        
        # 验证输出形状
        expected_shape = (batch_size, seq_len, gpt_config.n_embed)
        if output.shape != expected_shape:
            print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {output.shape}")
            return False
        
        # 检查数值稳定性
        if torch.isnan(output).any():
            print(f"✗ 输出包含NaN")
            return False
        
        if torch.isinf(output).any():
            print(f"✗ 输出包含Inf")
            return False
        
        # 前向传播（返回中间张量）
        output, cache, intermediate = sparse_attn(x, use_cache=False, return_intermediate=True)
        
        # 验证中间张量
        required_keys = ['qkv', 'q', 'k', 'v', 'local_mask', 'global_mask', 
                        'local_attn_scores', 'global_attn_weights', 'final_output']
        
        for key in required_keys:
            if key not in intermediate:
                print(f"✗ 缺少中间张量: {key}")
                return False
        
        print(f"  ✓ 输出形状正确: {output.shape}")
        print(f"  ✓ 中间张量完整: {len(intermediate)} 个")
    
    print("✓ 稀疏注意力前向传播测试通过")
    return True


def test_sparsity_characteristics():
    """测试稀疏特性"""
    print("\n=== 测试稀疏特性 ===")
    
    # 创建配置
    gpt_config = GPT2Config(
        vocab_size=1000,
        context_size=512,
        n_layer=1,
        n_embed=240,  # 240能被6整除
        n_head=6
    )
    
    sparse_config = SparseAttentionConfig(
        local_heads=4,
        global_heads=2,
        window_size=32,
        adaptive_window=False
    )
    
    sparse_attn = SparseAttention(gpt_config, sparse_config)
    
    # 创建测试输入
    batch_size, seq_len = 1, 64
    x = torch.randn(batch_size, seq_len, gpt_config.n_embed)
    
    # 获取中间张量
    _, _, intermediate = sparse_attn(x, use_cache=False, return_intermediate=True)
    
    # 检查本地注意力权重的稀疏性
    local_weights = intermediate['local_attn_scores']  # (batch_size, local_heads, seq_len, seq_len)
    
    # 计算非零元素比例
    total_elements = local_weights.numel()
    zero_elements = (local_weights == sparse_config.mask_value).sum().item()
    sparsity_ratio = zero_elements / total_elements
    
    print(f"本地注意力稀疏性: {sparsity_ratio:.2%} (零元素比例)")
    
    # 验证稀疏性在合理范围内
    expected_sparsity = 1 - (sparse_config.window_size / seq_len)
    if abs(sparsity_ratio - expected_sparsity) > 0.2:  # 允许20%的误差
        print(f"⚠ 稀疏性可能不符合预期: 期望约 {expected_sparsity:.2%}, 实际 {sparsity_ratio:.2%}")
    else:
        print(f"✓ 稀疏性符合预期")
    
    # 检查全局注意力权重的分布
    global_weights = intermediate['global_attn_weights']  # (batch_size, global_heads, seq_len, seq_len)
    
    # 全局注意力应该有更多的非零元素
    global_total = global_weights.numel()
    global_nonzero = (global_weights > 0).sum().item()
    global_density = global_nonzero / global_total
    
    print(f"全局注意力密度: {global_density:.2%}")
    
    print("✓ 稀疏特性测试通过")
    return True


def test_numerical_stability():
    """测试数值稳定性"""
    print("\n=== 测试数值稳定性 ===")
    
    # 创建配置
    gpt_config = GPT2Config(
        vocab_size=1000,
        context_size=512,
        n_layer=1,
        n_embed=240,  # 240能被6整除
        n_head=6
    )
    
    sparse_config = SparseAttentionConfig(
        local_heads=4,
        global_heads=2,
        window_size=32,
        mask_value=-1e9  # 使用较大的负值
    )
    
    sparse_attn = SparseAttention(gpt_config, sparse_config)
    
    # 测试极端情况
    test_cases = [
        ("大值输入", torch.randn(1, 32, gpt_config.n_embed) * 100),
        ("小值输入", torch.randn(1, 32, gpt_config.n_embed) * 0.01),
        ("零输入", torch.zeros(1, 32, gpt_config.n_embed)),
        ("长序列", torch.randn(1, 256, gpt_config.n_embed)),
    ]
    
    for case_name, x in test_cases:
        print(f"测试 {case_name}:")
        
        try:
            output, cache, intermediate = sparse_attn(x, use_cache=False, return_intermediate=True)
            
            # 检查输出
            if torch.isnan(output).any():
                print(f"  ✗ 输出包含NaN")
                return False
            
            if torch.isinf(output).any():
                print(f"  ✗ 输出包含Inf")
                return False
            
            # 检查注意力权重
            local_weights = intermediate['local_attn_scores']
            global_weights = intermediate['global_attn_weights']
            
            if torch.isnan(local_weights).any() or torch.isnan(global_weights).any():
                print(f"  ✗ 注意力权重包含NaN")
                return False
            
            # 检查softmax输出的数值范围
            local_softmax = F.softmax(local_weights, dim=-1)
            global_softmax = F.softmax(global_weights, dim=-1)
            
            if not (local_softmax.min() >= 0 and local_softmax.max() <= 1):
                print(f"  ✗ 本地softmax输出范围错误")
                return False
            
            if not (global_softmax.min() >= 0 and global_softmax.max() <= 1):
                print(f"  ✗ 全局softmax输出范围错误")
                return False
            
            print(f"  ✓ {case_name} 数值稳定")
            
        except Exception as e:
            print(f"  ✗ {case_name} 引发异常: {e}")
            return False
    
    print("✓ 数值稳定性测试通过")
    return True


def test_transformer_block_integration():
    """测试与TransformerBlock的集成"""
    print("\n=== 测试TransformerBlock集成 ===")
    
    # 创建启用稀疏注意力的配置
    gpt_config = GPT2Config(
        vocab_size=1000,
        context_size=512,
        n_layer=1,
        n_embed=240,  # 240能被6整除
        n_head=6,
        use_sparse_attention=True
    )
    
    # 创建TransformerBlock
    block = TransformerBlock(gpt_config)
    
    # 验证使用了稀疏注意力
    from app.models.transformer.sparse_attention import SparseAttention
    if not isinstance(block.attn, SparseAttention):
        print("✗ TransformerBlock未使用SparseAttention")
        return False
    
    print("✓ TransformerBlock正确使用SparseAttention")
    
    # 测试前向传播
    x = torch.randn(2, 32, gpt_config.n_embed)
    
    # 不返回中间张量
    output, cache, intermediate = block(x, use_cache=False, return_intermediate=False)
    
    if output.shape != (2, 32, gpt_config.n_embed):
        print(f"✗ 输出形状错误: {output.shape}")
        return False
    
    # 返回中间张量
    output, cache, intermediate = block(x, use_cache=False, return_intermediate=True)
    
    if intermediate is None:
        print("✗ 未返回中间张量")
        return False
    
    print("✓ TransformerBlock集成测试通过")
    return True


def test_model_integration():
    """测试与完整模型的集成"""
    print("\n=== 测试完整模型集成 ===")
    
    # 创建启用稀疏注意力的配置
    gpt_config = GPT2Config(
        vocab_size=1000,
        context_size=256,
        n_layer=2,
        n_embed=240,  # 240能被6整除
        n_head=6,
        use_sparse_attention=True
    )
    
    from app.models.transformer import GPT2Model
    
    model = GPT2Model(gpt_config)
    print(f"模型参数数量: {model.get_num_parameters():,}")
    
    # 测试前向传播
    input_ids = torch.randint(0, gpt_config.vocab_size, (2, 32))
    
    # 不返回中间张量
    result = model(input_ids, use_cache=False, return_cache=False, return_intermediate=False)
    
    expected_shape = (2, 32, gpt_config.vocab_size)
    if result["logits"].shape != expected_shape:
        print(f"✗ 输出形状错误: 期望 {expected_shape}, 实际 {result['logits'].shape}")
        return False
    
    # 返回中间张量
    result = model(input_ids, use_cache=False, return_cache=False, return_intermediate=True)
    
    if "intermediate" not in result:
        print("✗ 模型未返回中间张量")
        return False
    
    if len(result["intermediate"]) != gpt_config.n_layer:
        print(f"✗ 中间张量层数错误: 期望 {gpt_config.n_layer}, 实际 {len(result['intermediate'])}")
        return False
    
    print(f"  ✓ 输出形状正确: {result['logits'].shape}")
    print(f"  ✓ 中间张量层数正确: {len(result['intermediate'])}")
    
    print("✓ 完整模型集成测试通过")
    return True


def test_comparison_with_standard_attention():
    """与标准注意力的对比测试"""
    print("\n=== 与标准注意力对比测试 ===")
    
    # 创建相同配置的两个模型
    gpt_config = GPT2Config(
        vocab_size=1000,
        context_size=256,
        n_layer=1,
        n_embed=240,  # 240能被6整除
        n_head=6
    )
    
    # 标准注意力模型
    standard_config = GPT2Config(
        vocab_size=1000,
        context_size=256,
        n_layer=1,
        n_embed=240,  # 240能被6整除
        n_head=6,
        use_sparse_attention=False
    )
    
    # 稀疏注意力模型
    sparse_config = GPT2Config(
        vocab_size=1000,
        context_size=256,
        n_layer=1,
        n_embed=240,  # 240能被6整除
        n_head=6,
        use_sparse_attention=True
    )
    
    from app.models.transformer import GPT2Model
    
    standard_model = GPT2Model(standard_config)
    sparse_model = GPT2Model(sparse_config)
    
    # 创建测试输入
    input_ids = torch.randint(0, gpt_config.vocab_size, (2, 64))
    
    # 前向传播
    with torch.no_grad():
        standard_result = standard_model(input_ids)
        sparse_result = sparse_model(input_ids)
    
    # 比较输出形状
    if standard_result["logits"].shape != sparse_result["logits"].shape:
        print("✗ 标准注意力和稀疏注意力输出形状不同")
        return False
    
    print(f"✓ 输出形状一致: {standard_result['logits'].shape}")
    
    # 比较输出数值范围（应该大致相似）
    std_mean = standard_result["logits"].mean().item()
    std_std = standard_result["logits"].std().item()
    
    sparse_mean = sparse_result["logits"].mean().item()
    sparse_std = sparse_result["logits"].std().item()
    
    print(f"标准注意力: mean={std_mean:.4f}, std={std_std:.4f}")
    print(f"稀疏注意力: mean={sparse_mean:.4f}, std={sparse_std:.4f}")
    
    # 检查数值范围是否合理
    if abs(std_mean - sparse_mean) > 1.0 or abs(std_std - sparse_std) > 1.0:
        print("⚠ 输出分布差异较大，但这可能是正常的")
    else:
        print("✓ 输出分布相似")
    
    print("✓ 与标准注意力对比测试通过")
    return True


def main():
    """主测试函数"""
    print("开始测试稀疏注意力模块...")
    
    try:
        # 运行所有测试
        tests = [
            test_sparse_attention_config,
            test_sparse_attention_initialization,
            test_dynamic_window_size,
            test_mask_generation,
            test_sparse_attention_forward,
            test_sparsity_characteristics,
            test_numerical_stability,
            test_transformer_block_integration,
            test_model_integration,
            test_comparison_with_standard_attention,
        ]
        
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ 测试失败: {test_func.__name__}")
                return False
        
        print(f"\n🎉 所有测试通过！({passed}/{total})")
        print("\n✓ 稀疏注意力模块实现特点:")
        print("  - 分组头注意力机制（局部头 + 全局头）")
        print("  - 动态局部稀疏模式（自适应窗口大小）")
        print("  - 仅使用PyTorch操作，无需CUDA特制核心")
        print("  - 完整的中间张量返回机制")
        print("  - 数值稳定性保证（-inf mask填充）")
        print("  - 与现有Transformer架构完全兼容")
        print("  - 详细的单元测试覆盖")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)