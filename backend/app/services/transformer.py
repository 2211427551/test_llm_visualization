import numpy as np
from typing import List, Dict, Any, Tuple


class TransformerLayer:
    def __init__(self, n_embd: int, n_head: int, seed: int = 42):
        self.n_embd = n_embd
        self.n_head = n_head
        self.d_k = n_embd // n_head
        
        np.random.seed(seed)
        
        self.W_q = np.random.randn(n_embd, n_embd) * 0.02
        self.W_k = np.random.randn(n_embd, n_embd) * 0.02
        self.W_v = np.random.randn(n_embd, n_embd) * 0.02
        self.W_o = np.random.randn(n_embd, n_embd) * 0.02
        
        self.W_ffn1 = np.random.randn(n_embd, 4 * n_embd) * 0.02
        self.b_ffn1 = np.zeros(4 * n_embd)
        self.W_ffn2 = np.random.randn(4 * n_embd, n_embd) * 0.02
        self.b_ffn2 = np.zeros(n_embd)
        
        self.gamma_1 = np.ones(n_embd)
        self.beta_1 = np.zeros(n_embd)
        self.gamma_2 = np.ones(n_embd)
        self.beta_2 = np.zeros(n_embd)
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                   eps: float = 1e-5) -> Tuple[np.ndarray, Dict[str, Any]]:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + eps)
        output = gamma * x_norm + beta
        
        metadata = {
            "mean": float(np.mean(mean)),
            "variance": float(np.mean(variance)),
            "eps": eps
        }
        
        return output, metadata
    
    def multi_head_attention(self, x: np.ndarray) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        steps = []
        
        Q = np.dot(x, self.W_q)
        steps.append(("q_projection", Q, {"shape": Q.shape}))
        
        K = np.dot(x, self.W_k)
        steps.append(("k_projection", K, {"shape": K.shape}))
        
        V = np.dot(x, self.W_v)
        steps.append(("v_projection", V, {"shape": V.shape}))
        
        seq_len = x.shape[0]
        Q_heads = Q.reshape(seq_len, self.n_head, self.d_k)
        K_heads = K.reshape(seq_len, self.n_head, self.d_k)
        V_heads = V.reshape(seq_len, self.n_head, self.d_k)
        
        scores = np.matmul(Q_heads.transpose(1, 0, 2), K_heads.transpose(1, 2, 0)) / np.sqrt(self.d_k)
        steps.append(("attention_scores", scores.transpose(1, 2, 0), {
            "shape": scores.shape,
            "scaling_factor": float(np.sqrt(self.d_k))
        }))
        
        attention_weights = self._softmax(scores, axis=-1)
        steps.append(("attention_weights", attention_weights.transpose(1, 2, 0), {
            "shape": attention_weights.shape
        }))
        
        attention_output = np.matmul(attention_weights, V_heads.transpose(1, 0, 2))
        attention_output = attention_output.transpose(1, 0, 2).reshape(seq_len, self.n_embd)
        steps.append(("attention_output", attention_output, {"shape": attention_output.shape}))
        
        output = np.dot(attention_output, self.W_o)
        steps.append(("attention_projection", output, {"shape": output.shape}))
        
        return steps
    
    def feed_forward(self, x: np.ndarray) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        steps = []
        
        hidden = np.dot(x, self.W_ffn1) + self.b_ffn1
        steps.append(("ffn_hidden", hidden, {"shape": hidden.shape}))
        
        activated = np.maximum(0, hidden)
        steps.append(("ffn_relu", activated, {
            "shape": activated.shape,
            "activation": "relu"
        }))
        
        output = np.dot(activated, self.W_ffn2) + self.b_ffn2
        steps.append(("ffn_output", output, {"shape": output.shape}))
        
        return steps
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(self, x: np.ndarray) -> List[Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]]:
        all_steps = []
        
        norm1_out, norm1_meta = self.layer_norm(x, self.gamma_1, self.beta_1)
        all_steps.append(("layer_norm_1", x, norm1_out, norm1_meta))
        
        attn_steps = self.multi_head_attention(norm1_out)
        for step_type, output, metadata in attn_steps:
            all_steps.append((step_type, norm1_out, output, metadata))
        
        attn_output = attn_steps[-1][1]
        residual1 = x + attn_output
        all_steps.append(("residual_1", x, residual1, {"type": "residual_connection"}))
        
        norm2_out, norm2_meta = self.layer_norm(residual1, self.gamma_2, self.beta_2)
        all_steps.append(("layer_norm_2", residual1, norm2_out, norm2_meta))
        
        ffn_steps = self.feed_forward(norm2_out)
        for step_type, output, metadata in ffn_steps:
            all_steps.append((step_type, norm2_out, output, metadata))
        
        ffn_output = ffn_steps[-1][1]
        residual2 = residual1 + ffn_output
        all_steps.append(("residual_2", residual1, residual2, {"type": "residual_connection"}))
        
        return all_steps


class TransformerSimulator:
    def __init__(self, n_embd: int, n_layer: int, n_head: int, seed: int = 42):
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        
        self.layers = []
        for i in range(n_layer):
            layer_seed = seed + i
            self.layers.append(TransformerLayer(n_embd, n_head, layer_seed))
    
    def simulate(self, x: np.ndarray) -> List[Dict[str, Any]]:
        all_computation_steps = []
        current_input = x
        
        for layer_idx, layer in enumerate(self.layers):
            layer_steps = layer.forward(current_input)
            
            for step_type, input_data, output_data, metadata in layer_steps:
                step_info = {
                    "step_type": step_type,
                    "layer_index": layer_idx,
                    "input_data": input_data,
                    "output_data": output_data,
                    "metadata": metadata
                }
                all_computation_steps.append(step_info)
            
            current_input = layer_steps[-1][2]
        
        return all_computation_steps
