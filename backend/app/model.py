"""
Simple transformer-like model for forward tracing demonstration.
Implements a minimal architecture with embedding, attention, MoE, and output layers.
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class TensorTrace:
    """Container for traced tensor data with metadata."""
    name: str
    data: np.ndarray
    shape: List[int]
    dtype: str = "float32"
    truncated: bool = False
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    std_val: float = 0.0


@dataclass
class LayerTrace:
    """Container for all tensors traced during a layer's forward pass."""
    layer_id: int
    layer_name: str
    layer_type: str
    pre_activations: TensorTrace = None
    post_activations: TensorTrace = None
    weights: TensorTrace = None
    attention_data: Dict[str, Any] = None
    moe_data: Dict[str, Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleTransformerModel:
    """
    A minimal transformer-like model for demonstration purposes.
    Implements: Embedding -> Attention -> MoE -> Normalization -> Output
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_experts: int = 8,
        top_k_experts: int = 2,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        # Embedding layer
        self.embedding_matrix = np.random.randn(
            self.vocab_size, self.embedding_dim
        ) * 0.1
        
        # Attention parameters
        self.W_q = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        self.W_k = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        self.W_v = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        self.W_o = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        
        # MoE parameters
        self.gate_network = np.random.randn(self.embedding_dim, self.num_experts) * 0.1
        self.expert_networks = [
            np.random.randn(self.embedding_dim, self.hidden_dim) * 0.1
            for _ in range(self.num_experts)
        ]
        
        # Layer normalization parameters
        self.norm_gamma = np.ones(self.hidden_dim)
        self.norm_beta = np.zeros(self.hidden_dim)
        
        # Output projection
        self.output_projection = np.random.randn(
            self.hidden_dim, self.vocab_size
        ) * 0.1
        
    def forward(self, token_ids: List[int]) -> List[LayerTrace]:
        """
        Run forward pass and trace all intermediate computations.
        
        Args:
            token_ids: List of token IDs (integers from 0 to vocab_size-1)
            
        Returns:
            List of LayerTrace objects containing all intermediate tensors
        """
        traces = []
        num_tokens = len(token_ids)
        
        # Ensure token IDs are valid
        token_ids = [min(max(tid, 0), self.vocab_size - 1) for tid in token_ids]
        token_ids_array = np.array(token_ids)
        
        # Layer 0: Embedding
        embeddings = self.embedding_matrix[token_ids_array]  # [num_tokens, embedding_dim]
        traces.append(LayerTrace(
            layer_id=0,
            layer_name="Embedding Layer",
            layer_type="embedding",
            post_activations=self._trace_tensor("embeddings", embeddings),
            weights=self._trace_tensor("embedding_matrix", 
                                      self.embedding_matrix[:min(100, self.vocab_size)]),
            metadata={
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
                "num_tokens": num_tokens
            }
        ))
        
        # Layer 1: Multi-Head Attention
        attention_output, attention_data = self._multi_head_attention(embeddings)
        traces.append(LayerTrace(
            layer_id=1,
            layer_name="Multi-Head Attention",
            layer_type="attention",
            pre_activations=self._trace_tensor("attention_input", embeddings),
            post_activations=self._trace_tensor("attention_output", attention_output),
            attention_data=attention_data,
            metadata={
                "num_heads": self.num_heads,
                "head_dim": self.head_dim
            }
        ))
        
        # Layer 2: Mixture of Experts
        moe_output, moe_data = self._mixture_of_experts(attention_output)
        traces.append(LayerTrace(
            layer_id=2,
            layer_name="Mixture of Experts (MoE)",
            layer_type="moe",
            pre_activations=self._trace_tensor("moe_input", attention_output),
            post_activations=self._trace_tensor("moe_output", moe_output),
            moe_data=moe_data,
            metadata={
                "num_experts": self.num_experts,
                "top_k": self.top_k_experts
            }
        ))
        
        # Layer 3: Layer Normalization
        normalized = self._layer_norm(moe_output)
        traces.append(LayerTrace(
            layer_id=3,
            layer_name="Layer Normalization",
            layer_type="normalization",
            pre_activations=self._trace_tensor("norm_input", moe_output),
            post_activations=self._trace_tensor("norm_output", normalized),
            metadata={
                "hidden_dim": self.hidden_dim
            }
        ))
        
        # Layer 4: Output Projection
        logits = normalized @ self.output_projection
        traces.append(LayerTrace(
            layer_id=4,
            layer_name="Output Projection",
            layer_type="feedforward",
            pre_activations=self._trace_tensor("output_input", normalized),
            post_activations=self._trace_tensor("logits", logits),
            weights=self._trace_tensor("output_weights", self.output_projection),
            metadata={
                "vocab_size": self.vocab_size
            }
        ))
        
        # Layer 5: Softmax (final probabilities)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        traces.append(LayerTrace(
            layer_id=5,
            layer_name="Softmax Layer",
            layer_type="output",
            pre_activations=self._trace_tensor("softmax_input", logits),
            post_activations=self._trace_tensor("probabilities", probabilities),
            metadata={
                "vocab_size": self.vocab_size
            }
        ))
        
        return traces
    
    def _multi_head_attention(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Multi-head self-attention with tracing.
        
        Args:
            x: Input tensor [num_tokens, embedding_dim]
            
        Returns:
            Tuple of (output tensor, attention data dict)
        """
        num_tokens = x.shape[0]
        
        # Project to Q, K, V
        Q = x @ self.W_q  # [num_tokens, embedding_dim]
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Compute attention scores
        scores = (Q @ K.T) / np.sqrt(self.head_dim)  # [num_tokens, num_tokens]
        
        # Create causal mask and sparsity mask
        causal_mask = np.tril(np.ones((num_tokens, num_tokens)))
        sparsity_mask = (np.random.rand(num_tokens, num_tokens) > 0.3).astype(float)
        combined_mask = causal_mask * sparsity_mask
        
        # Apply mask
        scores_masked = scores * combined_mask + (1 - combined_mask) * (-1e9)
        
        # Softmax to get attention weights
        attention_weights = np.exp(scores_masked)
        attention_weights = attention_weights / (
            np.sum(attention_weights, axis=-1, keepdims=True) + 1e-9
        )
        
        # Apply attention to values
        attended = attention_weights @ V  # [num_tokens, embedding_dim]
        
        # Output projection
        output = attended @ self.W_o
        
        # Trace attention data
        attention_data = {
            "query_matrix": self._trace_tensor("Q", Q),
            "key_matrix": self._trace_tensor("K", K),
            "value_matrix": self._trace_tensor("V", V),
            "attention_scores": self._trace_tensor("attention_scores", attention_weights),
            "sparsity_mask": self._trace_tensor("sparsity_mask", combined_mask),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim
        }
        
        return output, attention_data
    
    def _mixture_of_experts(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Mixture of Experts layer with top-k gating.
        
        Args:
            x: Input tensor [num_tokens, embedding_dim]
            
        Returns:
            Tuple of (output tensor, MoE data dict)
        """
        num_tokens = x.shape[0]
        
        # Compute gating weights
        gate_logits = x @ self.gate_network  # [num_tokens, num_experts]
        gate_weights = np.exp(gate_logits) / np.sum(
            np.exp(gate_logits), axis=-1, keepdims=True
        )
        
        # Select top-k experts per token
        top_k_indices = np.argsort(gate_weights, axis=-1)[:, -self.top_k_experts:]
        
        # Create sparse gating weights (zero out non-selected experts)
        sparse_gate_weights = np.zeros_like(gate_weights)
        for i in range(num_tokens):
            sparse_gate_weights[i, top_k_indices[i]] = gate_weights[i, top_k_indices[i]]
        
        # Renormalize
        row_sums = np.sum(sparse_gate_weights, axis=-1, keepdims=True)
        sparse_gate_weights = sparse_gate_weights / (row_sums + 1e-9)
        
        # Compute expert outputs
        expert_outputs = []
        expert_activations = []
        for i, expert_net in enumerate(self.expert_networks):
            expert_out = np.maximum(0, x @ expert_net)  # ReLU activation
            expert_outputs.append(expert_out)
            expert_activations.append({
                "expert_id": i,
                "activations": self._trace_tensor(f"expert_{i}_activations", expert_out)
            })
        
        # Weighted combination of expert outputs
        output = np.zeros((num_tokens, self.hidden_dim))
        for i in range(num_tokens):
            for k in range(self.num_experts):
                output[i] += sparse_gate_weights[i, k] * expert_outputs[k][i]
        
        # Trace MoE data
        moe_data = {
            "gating_weights": self._trace_tensor("gate_weights", sparse_gate_weights),
            "selected_experts": top_k_indices.tolist(),
            "expert_activations": expert_activations,
            "num_experts": self.num_experts,
            "top_k": self.top_k_experts
        }
        
        return output, moe_data
    
    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return normalized * self.norm_gamma + self.norm_beta
    
    def _trace_tensor(
        self, name: str, tensor: np.ndarray, max_elements: int = 1000
    ) -> TensorTrace:
        """
        Create a traced tensor with statistics and truncation info.
        
        Args:
            name: Name of the tensor
            tensor: NumPy array to trace
            max_elements: Maximum elements to keep
            
        Returns:
            TensorTrace object
        """
        shape = list(tensor.shape)
        total_elements = tensor.size
        truncated = total_elements > max_elements
        
        # Compute statistics
        min_val = float(np.min(tensor))
        max_val = float(np.max(tensor))
        mean_val = float(np.mean(tensor))
        std_val = float(np.std(tensor))
        
        # Truncate if necessary
        if truncated:
            # For 2D tensors, keep a sample
            if len(shape) == 2:
                sample_rows = min(shape[0], int(np.sqrt(max_elements)))
                sample_cols = min(shape[1], int(np.sqrt(max_elements)))
                data = tensor[:sample_rows, :sample_cols]
            elif len(shape) == 1:
                data = tensor[:max_elements]
            else:
                data = tensor.flatten()[:max_elements]
        else:
            data = tensor
        
        return TensorTrace(
            name=name,
            data=data,
            shape=shape,
            dtype=str(tensor.dtype),
            truncated=truncated,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val
        )
