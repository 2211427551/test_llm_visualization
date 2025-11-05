#!/usr/bin/env python3
"""
简单的前向数据捕获测试脚本

不依赖pytest，直接测试核心功能
"""

import sys
import traceback

# 添加项目路径到Python路径
sys.path.insert(0, '/home/engine/project/backend')

try:
    import torch
    print("✓ PyTorch导入成功")
    TORCH_AVAILABLE = True
except ImportError:
    print("✗ PyTorch未安装，使用模拟模式")
    TORCH_AVAILABLE = False
    
    # 创建模拟环境
    from dataclasses import dataclass
    from typing import List, Dict, Any, Optional
    import json
    from datetime import datetime
    
    class MockTensor:
        def __init__(self, shape, dtype="float32"):
            self.shape = shape
            self.dtype = dtype
            self.device = "cpu"
        
        def size(self):
            return self.shape
        
        def numel(self):
            result = 1
            for dim in self.shape:
                result *= dim
            return result
        
        def detach(self):
            return MockTensor(self.shape, self.dtype)
        
        def cpu(self):
            return MockTensor(self.shape, self.dtype)
        
        def numpy(self):
            import numpy as np
            return np.zeros(self.shape)
        
        def max(self):
            return MockTensor([])
        
        def min(self):
            return MockTensor([])
        
        def mean(self):
            return MockTensor([])
        
        def item(self):
            return 1.0
        
        def sum(self, dim=None):
            if dim is None:
                return MockTensor([])
            else:
                new_shape = list(self.shape)
                if isinstance(dim, int):
                    new_shape[dim] = 1
                return MockTensor(new_shape)
        
        def any(self):
            return False
        
        def mean(self, dim=None):
            return MockTensor([])
        
        def __getitem__(self, key):
            if isinstance(key, slice):
                return MockTensor(list(self.shape)[key])
            return MockTensor([])
        
        def expansion(self, *sizes):
            return MockTensor(sizes)
        
        def unsqueeze(self, dim):
            new_shape = list(self.shape)
            new_shape.insert(dim, 1)
            return MockTensor(new_shape)
        
        def expand(self, *sizes):
            return MockTensor(list(sizes))
        
        def element_size(self):
            return 4  # float32
        
        @property
        def data(self):
            return self
        
        def normal_(self, mean=0.0, std=1.0):
            return self
        
        def zeros_(self):
            return self
        
        def ones_(self):
            return self
    
    class MockModule:
        def __init__(self):
            pass
        
        def parameters(self):
            return [MockTensor([100, 200]), MockTensor([200, 300])]
        
        def eval(self):
            pass
        
        def train(self):
            pass
        
        def zero_grad(self):
            pass
    
    # 模拟torch
    class MockTorch:
        Tensor = MockTensor
        
        @staticmethod
        def randint(low, high, size):
            return MockTensor(list(size))
        
        @staticmethod
        def randn(*size):
            return MockTensor(list(size))
        
        @staticmethod
        def empty(*size):
            return MockTensor(list(size))
        
        @staticmethod
        def allclose(a, b, atol=1e-6):
            return True
        
        @staticmethod
        def arange(n, device=None):
            return MockTensor([n])
        
        class device:
            def __init__(self, device_str):
                self.type = device_str.split(':')[0] if ':' in device_str else device_str
        
        @staticmethod
        def softmax(input, dim):
            return MockTensor(input.shape)
        
        @staticmethod
        def zeros_like(input):
            return MockTensor(input.shape)
        
        @staticmethod
        def var(input, dim=None):
            return MockTensor([])
        
        @staticmethod
        def topk(input, k, dim=-1, sorted=True):
            return MockTensor([*input.shape[:-1], k]), MockTensor([*input.shape[:-1], k])
        
        class no_grad:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        class nn:
            class Module:
                def __init__(self):
                    pass
                
                def eval(self):
                    pass
                
                def train(self):
                    pass
                
                def apply(self, fn):
                    # Apply function to all submodules
                    for attr_name in dir(self):
                        try:
                            attr = getattr(self, attr_name)
                            if hasattr(attr, 'apply') and callable(attr.apply):
                                attr.apply(fn)
                        except:
                            pass
                    return self
            
            class Linear(Module):
                def __init__(self, in_features, out_features, bias=True):
                    self.in_features = in_features
                    self.out_features = out_features
                    self.bias = bias
                    self.weight = MockTensor([out_features, in_features])
                    if bias:
                        self.bias = MockTensor([out_features])
                    else:
                        self.bias = None
            
            class Embedding(Module):
                def __init__(self, vocab_size, embed_dim):
                    self.vocab_size = vocab_size
                    self.embed_dim = embed_dim
                    self.weight = MockTensor([vocab_size, embed_dim])
            
            class LayerNorm(Module):
                def __init__(self, normalized_shape):
                    self.normalized_shape = normalized_shape
            
            class Dropout(Module):
                def __init__(self, p=0.5):
                    self.p = p
            
            class ModuleList(list):
                def __init__(self, modules):
                    super().__init__(modules)
            
            class functional:
                @staticmethod
                def mse_loss(input, target):
                    return MockTensor([])
                
                @staticmethod
                def softmax(input, dim):
                    return MockTensor(input.shape)
                
                @staticmethod
                def gelu(input):
                    return MockTensor(input.shape)
                
                @staticmethod
                def dropout(input, p=0.5, training=False):
                    return MockTensor(input.shape)
        
        class Size(tuple):
            def __getitem__(self, key):
                if isinstance(key, slice):
                    return list(self)[key]
                return super().__getitem__(key)
    
    # 替换torch导入
    sys.modules['torch'] = MockTorch()
    sys.modules['torch.nn'] = MockTorch.nn
    sys.modules['torch.nn.functional'] = MockTorch.nn.functional
    
    import torch

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试前向数据捕获基本功能...")
    
    try:
        from app.models.transformer.config import GPT2Config
        from app.models.transformer.model import GPT2Model
        from app.services.forward_capture import 数据捕获容器
        from app.schemas.forward_capture import 数据捕获配置
        
        print("✅ 成功导入所有模块")
        
        # 创建标准模型配置
        config = GPT2Config(
            vocab_size=1000,
            context_size=256,
            n_layer=2,  # 减少层数以加快测试
            n_head=8,
            n_embed=256,
            dropout=0.0,
            use_sparse_attention=False,
            use_moe=False,
        )
        
        print("✅ 成功创建模型配置")
        
        # 创建模型
        model = GPT2Model(config)
        model.eval()
        
        print("✅ 成功创建模型")
        
        # 创建输入数据
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        print(f"✅ 创建输入数据: {input_ids.shape}")
        
        # 配置数据捕获
        capture_config = 数据捕获配置(
            捕获嵌入数据=True,
            捕获注意力数据=True,
            捕获MoE数据=True,
            捕获最终输出=True,
            捕获张量值=False,  # 不捕获张量值以节省内存
        )
        
        print("✅ 成功创建捕获配置")
        
        # 执行带数据捕获的前向传播
        result = model.forward_with_capture(
            input_ids=input_ids,
            capture_config=capture_config
        )
        
        print("✅ 成功执行前向传播和数据捕获")
        
        # 验证结果结构
        assert "logits" in result, "缺少logits"
        assert "trajectory" in result, "缺少trajectory"
        assert "capture_stats" in result, "缺少capture_stats"
        
        print("✅ 结果结构验证通过")
        
        # 验证轨迹数据
        trajectory = result["trajectory"]
        assert trajectory.批次大小 == 2, f"批次大小错误: {trajectory.批次大小}"
        assert trajectory.序列长度 == 16, f"序列长度错误: {trajectory.序列长度}"
        assert trajectory.嵌入数据 is not None, "嵌入数据为空"
        assert trajectory.最终输出 is not None, "最终输出为空"
        
        print("✅ 轨迹数据验证通过")
        
        # 验证嵌入数据
        嵌入 = trajectory.嵌入数据
        assert 嵌入.输入序列长度 == 16, f"嵌入序列长度错误: {嵌入.输入序列长度}"
        assert 嵌入.批次大小 == 2, f"嵌入批次大小错误: {嵌入.批次大小}"
        assert 嵌入.嵌入维度 == config.n_embed, f"嵌入维度错误: {嵌入.嵌入维度}"
        
        print("✅ 嵌入数据验证通过")
        
        # 验证最终输出
        输出 = trajectory.最终输出
        assert 输出.Logits形状 == [2, 16, config.vocab_size], f"Logits形状错误: {输出.Logits形状}"
        assert 输出.词汇表大小 == config.vocab_size, f"词汇表大小错误: {输出.词汇表大小}"
        
        print("✅ 最终输出验证通过")
        
        # 测试JSON序列化
        json_data = trajectory.model_dump_json()
        assert len(json_data) > 0, "JSON序列化结果为空"
        
        print("✅ JSON序列化验证通过")
        
        print("\n🎉 所有基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
        return False

def test_sparse_attention():
    """测试稀疏注意力功能"""
    print("\n🧪 测试稀疏注意力数据捕获...")
    
    try:
        from app.models.transformer.config import GPT2Config
        from app.models.transformer.model import GPT2Model
        from app.schemas.forward_capture import 数据捕获配置
        
        # 创建稀疏注意力模型配置
        config = GPT2Config(
            vocab_size=1000,
            context_size=256,
            n_layer=2,
            n_head=12,  # 需要能被3整除
            n_embed=384,
            dropout=0.0,
            use_sparse_attention=True,
            use_moe=False,
        )
        
        model = GPT2Model(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        
        capture_config = 数据捕获配置(
            捕获嵌入数据=True,
            捕获注意力数据=True,
            捕获MoE数据=False,
            捕获最终输出=True,
        )
        
        result = model.forward_with_capture(
            input_ids=input_ids,
            capture_config=capture_config
        )
        
        trajectory = result["trajectory"]
        
        # 验证稀疏注意力数据
        assert len(trajectory.Transformer层数据) == config.n_layer
        
        for 层数据 in trajectory.Transformer层数据:
            if 层数据.注意力数据:
                注意力 = 层数据.注意力数据
                assert 注意力.局部注意力头数 > 0, "局部注意力头数应大于0"
                assert 注意力.全局注意力头数 > 0, "全局注意力头数应大于0"
                assert 注意力.注意力头数量 == config.n_head
                
                print(f"  第{层数据.层索引}层: {注意力.局部注意力头数}局部 + {注意力.全局注意力头数}全局头")
        
        print("✅ 稀疏注意力数据捕获测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 稀疏注意力测试失败: {e}")
        traceback.print_exc()
        return False

def test_moe():
    """测试MoE功能"""
    print("\n🧪 测试MoE数据捕获...")
    
    try:
        from app.models.transformer.config import GPT2Config
        from app.models.transformer.model import GPT2Model
        from app.schemas.forward_capture import 数据捕获配置
        
        # 创建MoE模型配置
        config = GPT2Config(
            vocab_size=1000,
            context_size=256,
            n_layer=2,
            n_head=8,
            n_embed=256,
            dropout=0.0,
            use_sparse_attention=False,
            use_moe=True,
            moe_num_experts=4,
            moe_top_k=2,
        )
        
        model = GPT2Model(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (2, 20))
        
        capture_config = 数据捕获配置(
            捕获嵌入数据=True,
            捕获注意力数据=True,
            捕获MoE数据=True,
            捕获最终输出=True,
        )
        
        result = model.forward_with_capture(
            input_ids=input_ids,
            capture_config=capture_config
        )
        
        trajectory = result["trajectory"]
        
        # 验证MoE数据
        moe_layers_found = 0
        for 层数据 in trajectory.Transformer层数据:
            if 层数据.MoE数据:
                moe_layers_found += 1
                MoE = 层数据.MoE数据
                assert MoE.专家总数 == config.moe_num_experts
                assert MoE.TopK值 == config.moe_top_k
                assert len(MoE.专家信息列表) == config.moe_num_experts
                
                print(f"  第{层数据.层索引}层: {MoE.专家总数}专家, TopK={MoE.TopK值}")
                print(f"    负载均衡损失: {MoE.负载均衡损失:.6f}")
        
        print(f"✅ MoE数据捕获测试通过！发现{moe_layers_found}个MoE层")
        return True
        
    except Exception as e:
        print(f"❌ MoE测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 前向数据捕获系统简单测试")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_sparse_attention,
        test_moe,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！前向数据捕获系统工作正常。")
        return 0
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    exit(main())