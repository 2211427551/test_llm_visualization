"""
MoE层API端点

提供REST API来演示MoE层的功能
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/engine/project/backend')

app = FastAPI(title="MoE Layer API", description="Mixture of Experts Layer API")

class MoEConfigRequest(BaseModel):
    """MoE配置请求"""
    n_embed: int = Field(default=768, description="嵌入维度")
    n_head: int = Field(default=12, description="注意力头数")
    use_moe: bool = Field(default=True, description="是否使用MoE")
    moe_num_experts: int = Field(default=8, description="专家数量")
    moe_top_k: int = Field(default=2, description="Top-k专家数量")
    moe_activation: str = Field(default="gelu", description="激活函数")
    moe_dropout: Optional[float] = Field(default=None, description="MoE dropout率")

class MoEConfigResponse(BaseModel):
    """MoE配置响应"""
    success: bool
    config: Dict[str, Any]
    message: str

@app.post("/moe/config/validate", response_model=MoEConfigResponse)
async def validate_moe_config(request: MoEConfigRequest):
    """验证MoE配置"""
    try:
        from app.models.transformer.config import GPT2Config
        
        # 创建配置
        config = GPT2Config(
            n_embed=request.n_embed,
            n_head=request.n_head,
            use_moe=request.use_moe,
            moe_num_experts=request.moe_num_experts,
            moe_top_k=request.moe_top_k,
            moe_activation=request.moe_activation,
            moe_dropout=request.moe_dropout
        )
        
        # 返回配置信息
        config_dict = {
            "n_embed": config.n_embed,
            "n_head": config.n_head,
            "use_moe": config.use_moe,
            "moe_num_experts": config.moe_num_experts,
            "moe_top_k": config.moe_top_k,
            "moe_activation": config.moe_activation,
            "moe_dropout": config.moe_dropout,
            "ffn_hidden_size": config.ffn_hidden_size,
            "head_dim": config.head_dim
        }
        
        return MoEConfigResponse(
            success=True,
            config=config_dict,
            message="MoE配置验证成功"
        )
        
    except ValueError as e:
        return MoEConfigResponse(
            success=False,
            config={},
            message=f"配置验证失败: {str(e)}"
        )
    except Exception as e:
        return MoEConfigResponse(
            success=False,
            config={},
            message=f"内部错误: {str(e)}"
        )

@app.get("/moe/features")
async def get_moe_features():
    """获取MoE功能特性"""
    features = {
        "core_features": [
            "独立的MoELayer实现",
            "Gating网络 (线性 + softmax)",
            "Top-k路由算法",
            "多个并行专家网络"
        ],
        "configuration_options": [
            "专家数量配置",
            "Top-k路由配置",
            "多种激活函数支持",
            "可配置dropout率"
        ],
        "supported_activations": ["gelu", "relu", "swish", "tanh"],
        "intermediate_data": [
            "gate_scores",
            "top_k_scores", 
            "top_k_indices",
            "expert_outputs",
            "final_output",
            "load_balance_loss"
        ],
        "integration_points": [
            "TransformerBlock集成",
            "与现有FFN无缝替换",
            "配置驱动的模式切换"
        ],
        "advanced_features": [
            "负载均衡损失",
            "专家使用统计",
            "梯度反向传播",
            "中间数据捕获"
        ]
    }
    
    return features

@app.get("/moe/architecture")
async def get_moe_architecture():
    """获取MoE架构信息"""
    architecture = {
        "components": {
            "MoELayer": {
                "description": "主要的MoE层实现",
                "methods": ["forward", "compute_load_balance_loss", "get_expert_usage_stats"]
            },
            "MoEExpert": {
                "description": "专家网络",
                "methods": ["forward"],
                "architecture": "Linear -> Activation -> Linear -> Dropout"
            },
            "GatingNetwork": {
                "description": "门控网络",
                "methods": ["forward"],
                "architecture": "Linear -> Softmax"
            }
        },
        "data_flow": [
            "输入 embedding",
            "Gating网络计算分数",
            "Top-k选择专家",
            "专家并行处理",
            "加权组合输出"
        ],
        "complexity": {
            "gating": "O(B × L × E × H)",
            "experts": "O(k × B × L × H²)",
            "total": "当 k << E 时有显著节省"
        }
    }
    
    return architecture

@app.get("/moe/testing")
async def get_testing_info():
    """获取测试信息"""
    testing_info = {
        "unit_tests": [
            "MoE专家网络前向传播",
            "Gating网络概率归一化",
            "Top-k路由正确性",
            "权重归一化验证",
            "梯度反向传播",
            "负载均衡损失计算",
            "专家使用统计",
            "不同配置组合",
            "TransformerBlock集成"
        ],
        "validation_tests": [
            "配置参数验证",
            "边界条件测试",
            "错误处理测试"
        ],
        "test_files": [
            "test_moe_unit.py - 完整单元测试",
            "test_moe_basic.py - 基本功能测试",
            "demo_moe_integration.py - 集成演示"
        ]
    }
    
    return testing_info

@app.get("/moe/examples")
async def get_examples():
    """获取使用示例"""
    examples = {
        "basic_usage": {
            "description": "基本MoE配置和使用",
            "code": """
from app.models.transformer.config import GPT2Config
from app.models.transformer.block import TransformerBlock

# 创建MoE配置
config = GPT2Config(
    n_embed=768,
    n_head=12,
    use_moe=True,
    moe_num_experts=8,
    moe_top_k=2
)

# 创建TransformerBlock
block = TransformerBlock(config)

# 前向传播
output, cache, intermediate = block(
    x, 
    use_cache=False, 
    return_intermediate=True
)
"""
        },
        "intermediate_data": {
            "description": "获取中间数据",
            "code": """
# 获取MoE中间数据
moe_data = intermediate['moe']
gate_scores = moe_data['gate_scores']
top_k_indices = moe_data['top_k_indices']
expert_outputs = moe_data['expert_outputs']
load_balance_loss = moe_data['load_balance_loss']
"""
        },
        "expert_stats": {
            "description": "专家使用统计",
            "code": """
# 获取专家使用统计
if isinstance(block.mlp, MoELayer):
    _, intermediate = block.mlp(x, return_intermediate=True)
    gate_scores = intermediate['gate_scores']
    stats = block.mlp.get_expert_usage_stats(gate_scores)
    print(f"专家使用频率: {stats['expert_usage']}")
    print(f"专家选择次数: {stats['expert_selections']}")
"""
        }
    }
    
    return examples

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "MoE Layer API",
        "version": "1.0.0",
        "description": "Mixture of Experts Layer Integration API",
        "endpoints": {
            "POST /moe/config/validate": "验证MoE配置",
            "GET /moe/features": "获取MoE功能特性",
            "GET /moe/architecture": "获取MoE架构信息",
            "GET /moe/testing": "获取测试信息",
            "GET /moe/examples": "获取使用示例"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)