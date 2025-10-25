import pytest
import numpy as np
from app.utils import (
    create_sparse_attention_mask,
    compute_moe_gating,
    truncate_tensor_data,
    validate_attention_sparsity
)


@pytest.mark.unit
class TestSparseAttentionMask:
    """Tests for sparse attention mask construction."""
    
    def test_mask_shape(self):
        """Test that mask has correct shape."""
        num_tokens = 10
        mask = create_sparse_attention_mask(num_tokens, seed=42)
        assert mask.shape == (num_tokens, num_tokens)
    
    def test_mask_binary(self):
        """Test that mask contains only binary values."""
        mask = create_sparse_attention_mask(10, seed=42)
        unique_values = np.unique(mask)
        assert set(unique_values).issubset({0, 1})
    
    def test_diagonal_always_one(self):
        """Test that diagonal elements are always 1 (self-attention)."""
        num_tokens = 10
        mask = create_sparse_attention_mask(num_tokens, sparsity_prob=0.9, seed=42)
        diagonal = np.diag(mask)
        assert np.all(diagonal == 1)
    
    def test_sparsity_pattern(self):
        """Test that sparsity approximately matches expected probability."""
        num_tokens = 100
        sparsity_prob = 0.3
        mask = create_sparse_attention_mask(num_tokens, sparsity_prob=sparsity_prob, seed=42)
        
        # Calculate actual sparsity (excluding diagonal which is always 1)
        off_diagonal_mask = mask.copy()
        np.fill_diagonal(off_diagonal_mask, 0)
        total_off_diagonal = num_tokens * num_tokens - num_tokens
        zeros_count = np.sum(off_diagonal_mask == 0)
        actual_sparsity = zeros_count / total_off_diagonal
        
        # Allow 10% tolerance
        assert abs(actual_sparsity - sparsity_prob) < 0.1
    
    def test_reproducibility(self):
        """Test that same seed produces same mask."""
        mask1 = create_sparse_attention_mask(10, seed=42)
        mask2 = create_sparse_attention_mask(10, seed=42)
        np.testing.assert_array_equal(mask1, mask2)
    
    def test_different_seeds_different_masks(self):
        """Test that different seeds produce different masks."""
        mask1 = create_sparse_attention_mask(10, seed=42)
        mask2 = create_sparse_attention_mask(10, seed=43)
        assert not np.array_equal(mask1, mask2)


@pytest.mark.unit
class TestMoEGating:
    """Tests for Mixture of Experts gating mechanism."""
    
    def test_gating_weights_shape(self):
        """Test that gating weights have correct shape."""
        token_embeddings = np.random.randn(10, 64)
        num_experts = 8
        gating_weights, _ = compute_moe_gating(token_embeddings, num_experts, seed=42)
        assert gating_weights.shape == (10, num_experts)
    
    def test_selected_experts_shape(self):
        """Test that selected experts have correct shape."""
        token_embeddings = np.random.randn(10, 64)
        num_experts = 8
        top_k = 2
        _, selected_experts = compute_moe_gating(
            token_embeddings, num_experts, top_k, seed=42
        )
        assert selected_experts.shape == (10, top_k)
    
    def test_exactly_top_k_experts_activated(self):
        """Test that exactly top-k experts are activated per token."""
        token_embeddings = np.random.randn(10, 64)
        num_experts = 8
        top_k = 2
        gating_weights, _ = compute_moe_gating(
            token_embeddings, num_experts, top_k, seed=42
        )
        
        # Count non-zero weights per token
        for token_idx in range(10):
            non_zero_count = np.sum(gating_weights[token_idx] > 0)
            assert non_zero_count == top_k
    
    def test_gates_sum_to_one(self):
        """Test that gating weights sum to 1 for each token."""
        token_embeddings = np.random.randn(10, 64)
        gating_weights, _ = compute_moe_gating(token_embeddings, seed=42)
        
        # Check that each row sums to 1
        row_sums = gating_weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10))
    
    def test_selected_experts_are_top_k(self):
        """Test that selected experts have highest gating weights."""
        token_embeddings = np.random.randn(10, 64)
        num_experts = 8
        top_k = 2
        
        # Use a seed to get reproducible results
        np.random.seed(42)
        raw_gating_weights = np.random.randn(10, num_experts)
        softmax_weights = np.exp(raw_gating_weights) / np.exp(raw_gating_weights).sum(axis=1, keepdims=True)
        
        gating_weights, selected_experts = compute_moe_gating(
            token_embeddings, num_experts, top_k, seed=42
        )
        
        for token_idx in range(10):
            selected = selected_experts[token_idx]
            top_k_indices = np.argsort(softmax_weights[token_idx])[-top_k:]
            # Selected experts should be in the top-k
            assert set(selected).issubset(set(top_k_indices))
    
    def test_gating_weights_non_negative(self):
        """Test that gating weights are non-negative."""
        token_embeddings = np.random.randn(10, 64)
        gating_weights, _ = compute_moe_gating(token_embeddings, seed=42)
        assert np.all(gating_weights >= 0)
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        token_embeddings = np.random.randn(10, 64)
        weights1, experts1 = compute_moe_gating(token_embeddings, seed=42)
        weights2, experts2 = compute_moe_gating(token_embeddings, seed=42)
        
        np.testing.assert_array_almost_equal(weights1, weights2)
        np.testing.assert_array_equal(experts1, experts2)
    
    def test_different_top_k(self):
        """Test gating with different top-k values."""
        token_embeddings = np.random.randn(10, 64)
        num_experts = 8
        
        for top_k in [1, 2, 4, 8]:
            gating_weights, selected_experts = compute_moe_gating(
                token_embeddings, num_experts, top_k, seed=42
            )
            assert selected_experts.shape == (10, top_k)
            
            for token_idx in range(10):
                non_zero_count = np.sum(gating_weights[token_idx] > 0)
                assert non_zero_count == top_k


@pytest.mark.unit
class TestTensorTruncation:
    """Tests for tensor data truncation."""
    
    def test_no_truncation_when_below_limit(self):
        """Test that small tensors are not truncated."""
        tensor = np.random.randn(10, 10)
        truncated, was_truncated = truncate_tensor_data(tensor, max_elements=200)
        assert not was_truncated
        np.testing.assert_array_equal(truncated, tensor)
    
    def test_truncation_when_above_limit(self):
        """Test that large tensors are truncated."""
        tensor = np.random.randn(100, 100)
        truncated, was_truncated = truncate_tensor_data(tensor, max_elements=1000)
        assert was_truncated
        assert truncated.size <= 1000
    
    def test_1d_tensor_truncation(self):
        """Test truncation of 1D tensors."""
        tensor = np.random.randn(5000)
        max_elements = 1000
        truncated, was_truncated = truncate_tensor_data(tensor, max_elements)
        assert was_truncated
        assert truncated.size == max_elements
        assert truncated.shape == (max_elements,)
    
    def test_2d_tensor_truncation(self):
        """Test truncation of 2D tensors."""
        tensor = np.random.randn(100, 100)
        max_elements = 400
        truncated, was_truncated = truncate_tensor_data(tensor, max_elements)
        assert was_truncated
        assert truncated.size <= max_elements
        expected_dim = int(np.sqrt(max_elements))
        assert truncated.shape == (expected_dim, expected_dim)
    
    def test_truncated_elements_preserved(self):
        """Test that truncated data preserves original elements."""
        tensor = np.random.randn(100, 100)
        truncated, _ = truncate_tensor_data(tensor, max_elements=400)
        expected_dim = int(np.sqrt(400))
        np.testing.assert_array_equal(truncated, tensor[:expected_dim, :expected_dim])


@pytest.mark.unit
class TestAttentionSparsityValidation:
    """Tests for attention sparsity validation."""
    
    def test_valid_sparse_attention(self):
        """Test validation passes for correctly masked attention."""
        num_tokens = 10
        attention_scores = np.random.rand(num_tokens, num_tokens)
        sparsity_mask = np.random.randint(0, 2, (num_tokens, num_tokens))
        
        # Apply mask to attention scores
        masked_attention = attention_scores * sparsity_mask
        
        assert validate_attention_sparsity(masked_attention, sparsity_mask)
    
    def test_invalid_sparse_attention(self):
        """Test validation fails for incorrectly masked attention."""
        num_tokens = 10
        attention_scores = np.ones((num_tokens, num_tokens))
        sparsity_mask = np.zeros((num_tokens, num_tokens))
        
        # Attention has values where mask is zero
        assert not validate_attention_sparsity(attention_scores, sparsity_mask)
    
    def test_full_attention_with_full_mask(self):
        """Test validation with no sparsity (all connections active)."""
        num_tokens = 10
        attention_scores = np.random.rand(num_tokens, num_tokens)
        sparsity_mask = np.ones((num_tokens, num_tokens))
        
        assert validate_attention_sparsity(attention_scores, sparsity_mask)
    
    def test_zero_attention_with_zero_mask(self):
        """Test validation with fully sparse attention."""
        num_tokens = 10
        attention_scores = np.zeros((num_tokens, num_tokens))
        sparsity_mask = np.zeros((num_tokens, num_tokens))
        
        assert validate_attention_sparsity(attention_scores, sparsity_mask)
