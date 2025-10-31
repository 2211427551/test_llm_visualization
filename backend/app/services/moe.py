import numpy as np
from typing import List, Dict, Any, Tuple
from app.utils.config import MoEConfig
from app.models.response import ExpertInfo, MoERouting


class Expert:
    """单个专家网络"""
    def __init__(self, expert_id: int, n_embd: int, seed: int = 42):
        self.expert_id = expert_id
        self.n_embd = n_embd
        
        np.random.seed(seed + expert_id)
        
        # 两层前馈网络
        self.W1 = np.random.randn(n_embd, 4 * n_embd) * 0.02
        self.b1 = np.zeros(4 * n_embd)
        self.W2 = np.random.randn(4 * n_embd, n_embd) * 0.02
        self.b2 = np.zeros(n_embd)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """专家前向传播"""
        hidden = np.dot(x, self.W1) + self.b1
        activated = np.maximum(0, hidden)  # ReLU
        output = np.dot(activated, self.W2) + self.b2
        return output


class MoELayer:
    """MoE (Mixture of Experts) 层"""
    def __init__(self, n_embd: int, config: MoEConfig, seed: int = 42):
        self.n_embd = n_embd
        self.config = config
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        
        # 创建专家
        self.experts = [
            Expert(i, n_embd, seed + i) 
            for i in range(self.n_experts)
        ]
        
        # 门控网络
        np.random.seed(seed)
        self.W_gate = np.random.randn(n_embd, self.n_experts) * 0.02
        self.b_gate = np.zeros(self.n_experts)
    
    def compute_gate_logits(self, x: np.ndarray) -> np.ndarray:
        """计算门控logits"""
        gate_logits = np.dot(x, self.W_gate) + self.b_gate
        
        # 添加噪声
        if self.config.gate_noise > 0:
            noise = np.random.normal(0, self.config.gate_noise, gate_logits.shape)
            gate_logits += noise
        
        return gate_logits
    
    def select_top_k_experts(self, gate_logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """选择top-k专家"""
        top_k_logits, top_k_indices = np.topk(gate_logits, self.top_k)
        return top_k_logits, top_k_indices
    
    def combine_expert_outputs(self, expert_outputs: List[np.ndarray], 
                             gate_weights: np.ndarray) -> np.ndarray:
        """组合专家输出"""
        if not expert_outputs:
            return np.zeros(self.n_embd)
        
        # 加权平均
        combined_output = np.zeros_like(expert_outputs[0])
        for i, output in enumerate(expert_outputs):
            combined_output += gate_weights[i] * output
        
        return combined_output
    
    def forward(self, x: np.ndarray):
        """MoE前向传播"""
        steps = []
        
        # 计算门控logits
        gate_logits = self.compute_gate_logits(x)
        steps.append(("moe_gate_logits", gate_logits, {"shape": gate_logits.shape}))
        
        # 应用softmax得到门控概率
        gate_probs = self._softmax(gate_logits)
        steps.append(("moe_gate_probs", gate_probs, {"shape": gate_probs.shape}))
        
        # 选择top-k专家
        top_k_logits, top_k_indices = self.select_top_k_experts(gate_logits)
        top_k_probs = gate_probs[top_k_indices]
        
        # 归一化top-k概率
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        # 运行选中的专家
        expert_outputs = []
        expert_infos = []
        
        for i, expert_idx in enumerate(top_k_indices):
            expert = self.experts[expert_idx]
            expert_output = expert.forward(x)
            expert_outputs.append(expert_output)
            
            expert_info = ExpertInfo(
                expert_id=int(expert_idx),
                gate_probability=float(top_k_probs[i]),
                output=expert_output.tolist()
            )
            expert_infos.append(expert_info)
            
            steps.append((
                f"expert_{expert_idx}_output", 
                expert_output, 
                {"gate_prob": float(top_k_probs[i])}
            ))
        
        # 组合专家输出
        combined_output = self.combine_expert_outputs(expert_outputs, top_k_probs)
        steps.append(("moe_combined_output", combined_output, {"num_experts": len(expert_outputs)}))
        
        # 创建MoE路由信息
        moe_routing = MoERouting(
            top_k_experts=expert_infos,
            gate_logits=gate_logits.tolist(),
            combined_output=combined_output.tolist()
        )
        
        return combined_output, moe_routing, steps
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax函数"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MoESimulator:
    """MoE模拟器"""
    def __init__(self, n_embd: int, config: MoEConfig, n_layers: int = 1, seed: int = 42):
        self.n_embd = n_embd
        self.config = config
        self.n_layers = n_layers
        
        if config.enabled:
            self.moe_layers = [
                MoELayer(n_embd, config, seed + i) 
                for i in range(n_layers)
            ]
        else:
            self.moe_layers = []
    
    def simulate_layer(self, x: np.ndarray, layer_idx: int):
        """模拟单个MoE层"""
        if not self.config.enabled or layer_idx >= len(self.moe_layers):
            # 如果未启用MoE或层索引超出范围，返回简单的FFN
            simple_output = self._simple_ffn(x)
            return simple_output, None, [("simple_ffn", simple_output, {"type": "standard"})]
        
        moe_layer = self.moe_layers[layer_idx]
        return moe_layer.forward(x)
    
    def _simple_ffn(self, x: np.ndarray) -> np.ndarray:
        """简单的前馈网络（当MoE未启用时）"""
        # 模拟标准FFN
        hidden = np.dot(x, np.random.randn(self.n_embd, 4 * self.n_embd) * 0.02)
        activated = np.maximum(0, hidden)
        output = np.dot(activated, np.random.randn(4 * self.n_embd, self.n_embd) * 0.02)
        return output