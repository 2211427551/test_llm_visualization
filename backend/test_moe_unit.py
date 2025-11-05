"""
MoE层单元测试

测试Mixture of Experts层的各种功能：
1. Top-k路由正确性
2. 权重归一化
3. 梯度反向传播
4. 负载均衡
5. 中间数据捕获
"""

import torch
import torch.nn as nn
from app.models.transformer.config import GPT2Config
from app.models.transformer.moe import MoELayer, MoEExpert, GatingNetwork
from app.models.transformer.block import TransformerBlock


class TestMoEExpert:
    """测试MoE专家网络"""
    
    def test_expert_forward(self):
        """测试专家网络前向传播"""
        config = GPT2Config(n_embed=256, ffn_hidden_multiplier=4)
        expert = MoEExpert(config)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.n_embed)
        
        output = expert(x)
        
        # 验证输出形状
        assert output.shape == (batch_size, seq_len, config.n_embed)
        
        # 验证输出不是NaN或Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_expert_different_activations(self):
        """测试不同激活函数"""
        activations = ["gelu", "relu", "swish", "tanh"]
        
        for activation in activations:
            config = GPT2Config(
                n_embed=128, 
                ffn_hidden_multiplier=4,
                moe_activation=activation
            )
            expert = MoEExpert(config)
            
            x = torch.randn(1, 5, config.n_embed)
            output = expert(x)
            
            assert output.shape == (1, 5, config.n_embed)
            assert not torch.isnan(output).any()


class TestGatingNetwork:
    """测试Gating网络"""
    
    def test_gating_forward(self):
        """测试门控网络前向传播"""
        n_embed = 256
        num_experts = 8
        
        gating = GatingNetwork(n_embed, num_experts)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, n_embed)
        
        gate_scores = gating(x)
        
        # 验证输出形状
        assert gate_scores.shape == (batch_size, seq_len, num_experts)
        
        # 验证概率归一化（每行和为1）
        row_sums = gate_scores.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
        
        # 验证概率范围在[0, 1]
        assert (gate_scores >= 0).all()
        assert (gate_scores <= 1).all()


class TestMoELayer:
    """测试MoE层"""
    
    def test_moe_forward(self):
        """测试MoE层前向传播"""
        config = GPT2Config(n_embed=256, ffn_hidden_multiplier=4)
        moe = MoELayer(config, num_experts=4, top_k=2)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.n_embed)
        
        output, intermediate = moe(x, return_intermediate=True)
        
        # 验证输出形状
        assert output.shape == (batch_size, seq_len, config.n_embed)
        
        # 验证中间数据
        assert intermediate is not None
        assert 'gate_scores' in intermediate
        assert 'top_k_scores' in intermediate
        assert 'top_k_indices' in intermediate
        assert 'expert_outputs' in intermediate
        assert 'final_output' in intermediate
        assert 'load_balance_loss' in intermediate
        
        # 验证门控分数形状
        assert intermediate['gate_scores'].shape == (batch_size, seq_len, 4)
        
        # 验证top-k分数和索引形状
        assert intermediate['top_k_scores'].shape == (batch_size, seq_len, 2)
        assert intermediate['top_k_indices'].shape == (batch_size, seq_len, 2)
    
    def test_top_k_routing(self):
        """测试Top-k路由正确性"""
        config = GPT2Config(n_embed=128)
        num_experts = 4
        top_k = 2
        
        moe = MoELayer(config, num_experts=num_experts, top_k=top_k)
        
        batch_size, seq_len = 1, 3
        x = torch.randn(batch_size, seq_len, config.n_embed)
        
        output, intermediate = moe(x, return_intermediate=True)
        
        gate_scores = intermediate['gate_scores']
        top_k_scores = intermediate['top_k_scores']
        top_k_indices = intermediate['top_k_indices']
        
        # 验证top-k分数确实是最大的k个值
        for i in range(seq_len):
            expected_scores, expected_indices = torch.topk(
                gate_scores[0, i], k=top_k, sorted=True
            )
            assert torch.allclose(top_k_scores[0, i], expected_scores, atol=1e-6)
            assert torch.equal(top_k_indices[0, i], expected_indices)
    
    def test_weight_normalization(self):
        """测试权重归一化"""
        config = GPT2Config(n_embed=128)
        moe = MoELayer(config, num_experts=4, top_k=2)
        
        x = torch.randn(1, 5, config.n_embed)
        _, intermediate = moe(x, return_intermediate=True)
        
        top_k_scores = intermediate['top_k_scores']
        
        # 验证每行的top-k分数和为1
        row_sums = top_k_scores.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    
    def test_gradient_flow(self):
        """测试梯度反向传播"""
        config = GPT2Config(n_embed=128)
        moe = MoELayer(config, num_experts=4, top_k=2)
        
        x = torch.randn(2, 3, config.n_embed, requires_grad=True)
        target = torch.randn(2, 3, config.n_embed)
        
        output, _ = moe(x, return_intermediate=True)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # 验证输入梯度存在
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # 验证模型参数梯度存在
        for param in moe.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_load_balance_loss(self):
        """测试负载均衡损失"""
        config = GPT2Config(n_embed=128)
        moe = MoELayer(config, num_experts=4, top_k=2)
        
        x = torch.randn(2, 10, config.n_embed)
        _, intermediate = moe(x, return_intermediate=True)
        
        load_balance_loss = intermediate['load_balance_loss']
        
        # 验证损失是标量且非负
        assert load_balance_loss.dim() == 0
        assert load_balance_loss >= 0
    
    def test_expert_usage_stats(self):
        """测试专家使用统计"""
        config = GPT2Config(n_embed=128)
        moe = MoELayer(config, num_experts=4, top_k=2)
        
        x = torch.randn(2, 10, config.n_embed)
        _, intermediate = moe(x, return_intermediate=True)
        
        gate_scores = intermediate['gate_scores']
        stats = moe.get_expert_usage_stats(gate_scores)
        
        # 验证统计信息
        assert 'expert_usage' in stats
        assert 'expert_selections' in stats
        assert 'usage_std' in stats
        assert 'selections_std' in stats
        
        # 验证形状
        assert stats['expert_usage'].shape == (4,)
        assert stats['expert_selections'].shape == (4,)
    
    def test_different_configurations(self):
        """测试不同配置的MoE层"""
        configs = [
            {'num_experts': 2, 'top_k': 1},
            {'num_experts': 8, 'top_k': 2},
            {'num_experts': 16, 'top_k': 4},
        ]
        
        for config_dict in configs:
            config = GPT2Config(n_embed=128)
            moe = MoELayer(config, **config_dict)
            
            x = torch.randn(1, 5, config.n_embed)
            output, _ = moe(x)
            
            assert output.shape == (1, 5, config.n_embed)


class TestTransformerBlockWithMoE:
    """测试集成MoE的TransformerBlock"""
    
    def test_block_with_moe(self):
        """测试使用MoE的TransformerBlock"""
        config = GPT2Config(
            n_embed=256,
            n_head=8,
            n_layer=1,
            use_moe=True,
            moe_num_experts=4,
            moe_top_k=2
        )
        
        block = TransformerBlock(config)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.n_embed)
        
        output, cache, intermediate = block(
            x, 
            use_cache=False, 
            return_intermediate=True
        )
        
        # 验证输出形状
        assert output.shape == (batch_size, seq_len, config.n_embed)
        
        # 验证MoE中间数据
        assert intermediate is not None
        assert 'moe' in intermediate
        assert 'gate_scores' in intermediate['moe']
        assert 'top_k_indices' in intermediate['moe']
    
    def test_block_without_moe(self):
        """测试不使用MoE的TransformerBlock"""
        config = GPT2Config(
            n_embed=256,
            n_head=8,
            n_layer=1,
            use_moe=False
        )
        
        block = TransformerBlock(config)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.n_embed)
        
        output, cache, intermediate = block(
            x, 
            use_cache=False, 
            return_intermediate=True
        )
        
        # 验证输出形状
        assert output.shape == (batch_size, seq_len, config.n_embed)
        
        # 验证没有MoE中间数据
        if intermediate is not None:
            assert 'moe' not in intermediate
    
    def test_moe_vs_standard_ffn(self):
        """比较MoE和标准FFN的输出差异"""
        config_moe = GPT2Config(
            n_embed=128,
            n_head=4,
            use_moe=True,
            moe_num_experts=4,
            moe_top_k=2
        )
        
        config_standard = GPT2Config(
            n_embed=128,
            n_head=4,
            use_moe=False
        )
        
        block_moe = TransformerBlock(config_moe)
        block_standard = TransformerBlock(config_standard)
        
        x = torch.randn(1, 5, config_moe.n_embed)
        
        output_moe, _, _ = block_moe(x, return_intermediate=True)
        output_standard, _, _ = block_standard(x, return_intermediate=True)
        
        # 输出应该不同（MoE和标准FFN有不同的计算路径）
        assert not torch.allclose(output_moe, output_standard, atol=1e-6)
        
        # 但输出形状应该相同
        assert output_moe.shape == output_standard.shape


if __name__ == "__main__":
    # 运行基本测试
    print("开始运行MoE单元测试...")
    
    test_expert = TestMoEExpert()
    test_expert.test_expert_forward()
    test_expert.test_expert_different_activations()
    print("✓ MoE专家网络测试通过")
    
    test_gating = TestGatingNetwork()
    test_gating.test_gating_forward()
    print("✓ Gating网络测试通过")
    
    test_moe = TestMoELayer()
    test_moe.test_moe_forward()
    test_moe.test_top_k_routing()
    test_moe.test_weight_normalization()
    test_moe.test_gradient_flow()
    test_moe.test_load_balance_loss()
    test_moe.test_expert_usage_stats()
    test_moe.test_different_configurations()
    print("✓ MoE层测试通过")
    
    test_block = TestTransformerBlockWithMoE()
    test_block.test_block_with_moe()
    test_block.test_block_without_moe()
    test_block.test_moe_vs_standard_ffn()
    print("✓ TransformerBlock集成测试通过")
    
    print("\n🎉 所有MoE测试通过！")