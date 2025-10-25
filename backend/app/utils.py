import numpy as np
from typing import Tuple


def create_sparse_attention_mask(
    num_tokens: int,
    sparsity_prob: float = 0.3,
    seed: int = None
) -> np.ndarray:
    """
    Create a sparse attention mask for attention mechanism.
    
    Args:
        num_tokens: Number of tokens in the sequence
        sparsity_prob: Probability of pruning an attention connection (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Binary mask of shape [num_tokens, num_tokens] where 1 = keep, 0 = prune
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random mask: 1 where we keep connections, 0 where we prune
    mask = (np.random.rand(num_tokens, num_tokens) > sparsity_prob).astype(int)
    
    # Ensure each token attends to at least itself (diagonal is always 1)
    np.fill_diagonal(mask, 1)
    
    return mask


def compute_moe_gating(
    token_embeddings: np.ndarray,
    num_experts: int = 8,
    top_k: int = 2,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MoE gating weights and select top-k experts for each token.
    
    Args:
        token_embeddings: Token embeddings of shape [num_tokens, embedding_dim]
        num_experts: Number of expert networks
        top_k: Number of experts to activate per token
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of:
        - gating_weights: Normalized gating weights [num_tokens, num_experts]
        - selected_experts: Indices of top-k experts [num_tokens, top_k]
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_tokens = token_embeddings.shape[0]
    
    # Simulate gating network with random projection
    gating_logits = np.random.randn(num_tokens, num_experts)
    
    # Apply softmax to get normalized gating weights
    gating_weights = np.exp(gating_logits) / np.exp(gating_logits).sum(axis=1, keepdims=True)
    
    # Select top-k experts per token
    selected_experts = np.argsort(gating_weights, axis=1)[:, -top_k:]
    
    # Zero out weights for non-selected experts
    gated_weights = np.zeros_like(gating_weights)
    for i in range(num_tokens):
        gated_weights[i, selected_experts[i]] = gating_weights[i, selected_experts[i]]
    
    # Re-normalize the selected expert weights to sum to 1
    row_sums = gated_weights.sum(axis=1, keepdims=True)
    gated_weights = gated_weights / row_sums
    
    return gated_weights, selected_experts


def truncate_tensor_data(
    tensor: np.ndarray,
    max_elements: int = 1000
) -> Tuple[np.ndarray, bool]:
    """
    Truncate tensor data if it exceeds maximum element count.
    
    Args:
        tensor: Input tensor to potentially truncate
        max_elements: Maximum number of elements to keep
    
    Returns:
        Tuple of (truncated_tensor, was_truncated)
    """
    total_elements = tensor.size
    
    if total_elements <= max_elements:
        return tensor, False
    
    # Calculate new shape to maintain aspect ratio as much as possible
    shape = tensor.shape
    if len(shape) == 1:
        return tensor[:max_elements], True
    elif len(shape) == 2:
        # For 2D tensors, try to keep square-ish
        max_dim = int(np.sqrt(max_elements))
        return tensor[:max_dim, :max_dim], True
    else:
        # For higher dimensions, flatten and truncate
        return tensor.flatten()[:max_elements], True


def validate_attention_sparsity(
    attention_scores: np.ndarray,
    sparsity_mask: np.ndarray
) -> bool:
    """
    Validate that attention scores respect the sparsity mask.
    
    Args:
        attention_scores: Attention scores of shape [num_tokens, num_tokens]
        sparsity_mask: Binary mask of shape [num_tokens, num_tokens]
    
    Returns:
        True if all masked positions have zero attention score
    """
    masked_positions = sparsity_mask == 0
    return np.all(attention_scores[masked_positions] == 0)
