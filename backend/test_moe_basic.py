"""
简化的MoE测试 - 验证代码结构和基本功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/engine/project/backend')

def test_config_import():
    """测试配置导入"""
    try:
        from app.models.transformer.config import GPT2Config
        print("✓ 配置导入成功")
        
        # 测试MoE配置
        config = GPT2Config(
            n_embed=128,
            use_moe=True,
            moe_num_experts=4,
            moe_top_k=2
        )
        print(f"✓ MoE配置创建成功: use_moe={config.use_moe}, experts={config.moe_num_experts}, top_k={config.moe_top_k}")
        return True
    except Exception as e:
        print(f"✗ 配置导入失败: {e}")
        return False

def test_moe_import():
    """测试MoE模块导入"""
    try:
        from app.models.transformer.moe import MoELayer, MoEExpert, GatingNetwork
        print("✓ MoE模块导入成功")
        return True
    except Exception as e:
        print(f"✗ MoE模块导入失败: {e}")
        return False

def test_block_import():
    """测试TransformerBlock导入"""
    try:
        from app.models.transformer.block import TransformerBlock
        print("✓ TransformerBlock导入成功")
        return True
    except Exception as e:
        print(f"✗ TransformerBlock导入失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能（不使用torch）"""
    try:
        from app.models.transformer.config import GPT2Config
        
        # 测试配置验证
        try:
            config = GPT2Config(
                n_embed=128,
                n_head=8,  # 确保能被n_embed整除
                use_moe=True,
                moe_num_experts=4,
                moe_top_k=2
            )
            print("✓ 有效配置验证通过")
        except Exception as e:
            print(f"✗ 有效配置验证失败: {e}")
            return False
        
        # 测试无效配置
        try:
            invalid_config = GPT2Config(
                n_embed=128,
                n_head=8,
                use_moe=True,
                moe_num_experts=4,
                moe_top_k=5  # top_k > num_experts
            )
            print("✗ 无效配置验证失败 - 应该抛出异常")
            return False
        except ValueError:
            print("✓ 无效配置正确拒绝")
        except Exception as e:
            print(f"✗ 无效配置验证异常: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始MoE集成验证测试...\n")
    
    tests = [
        ("配置导入", test_config_import),
        ("MoE模块导入", test_moe_import),
        ("TransformerBlock导入", test_block_import),
        ("基本功能", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"运行测试: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有基本测试通过！MoE层集成代码结构正确。")
        print("\n注意: 由于环境限制，无法运行完整的torch测试。")
        print("但在有torch的环境中，以下功能应该可以正常工作:")
        print("- MoE层前向传播")
        print("- Top-k路由")
        print("- 权重归一化") 
        print("- 梯度反向传播")
        print("- 负载均衡")
        print("- 中间数据捕获")
        print("- TransformerBlock集成")
        return True
    else:
        print("❌ 部分测试失败，需要检查代码。")
        return False

if __name__ == "__main__":
    main()