import pytest
import numpy as np
from app.utils import truncate_tensor_data


@pytest.mark.regression
class TestSerializationTruncation:
    """Regression tests for intermediate serialization truncation behavior."""
    
    def test_no_tensor_exceeds_max_elements_after_truncation(self):
        """Test that truncated tensors never exceed defined max elements."""
        max_elements = 1000
        
        # Test various tensor sizes
        test_shapes = [
            (100, 100),
            (50, 50, 4),
            (200, 200),
            (10000,),
            (32, 64, 8)
        ]
        
        for shape in test_shapes:
            tensor = np.random.randn(*shape)
            truncated, was_truncated = truncate_tensor_data(tensor, max_elements)
            
            # Ensure truncated tensor never exceeds limit
            assert truncated.size <= max_elements, (
                f"Truncated tensor of shape {truncated.shape} has {truncated.size} elements, "
                f"exceeding max of {max_elements}"
            )
            
            # Verify truncation flag is set correctly
            if tensor.size > max_elements:
                assert was_truncated, f"Expected truncation for tensor of size {tensor.size}"
    
    def test_large_model_response_truncation(self, client, large_model_request):
        """Test that large model responses trigger truncation."""
        response = client.post("/model/forward", json=large_model_request)
        assert response.status_code == 200
        data = response.json()
        
        # Should be marked as truncated
        assert data["truncated"] is True
        assert data["warnings"] is not None
        
        # Check that layers have truncated flag
        for step in data["steps"]:
            layer_data = step["layerData"]
            assert layer_data.get("truncated") is True
    
    def test_truncated_layers_have_none_or_small_tensors(self, client, large_model_request):
        """Test that truncated layers have None or small tensor data."""
        response = client.post("/model/forward", json=large_model_request)
        data = response.json()
        
        for step in data["steps"]:
            layer_data = step["layerData"]
            
            if layer_data.get("truncated"):
                # Activations should be None or small
                activations = layer_data.get("activations")
                assert activations is None or len(activations) <= 10
                
                # Weights should be None or small
                weights = layer_data.get("weights")
                assert weights is None or len(weights) <= 10
                
                # Attention data should be None for truncated layers
                attention_data = layer_data.get("attentionData")
                assert attention_data is None
                
                # MoE data should be None for truncated layers
                moe_data = layer_data.get("moeData")
                assert moe_data is None
    
    def test_small_model_response_no_truncation(self, client, sample_model_request):
        """Test that small model responses do not trigger truncation."""
        response = client.post("/model/forward", json=sample_model_request)
        data = response.json()
        
        # Should not be marked as truncated
        assert data.get("truncated") is False or data.get("truncated") is None
        
        # Check that layers have data
        for step in data["steps"]:
            layer_data = step["layerData"]
            assert layer_data.get("truncated") is False or layer_data.get("truncated") is None
    
    def test_truncation_threshold_at_100_tokens(self, client):
        """Test that truncation is triggered at exactly 100 tokens."""
        # 100 tokens - should not truncate (use single char words to stay under 500 char limit)
        text_100 = " ".join(["a"] * 100)
        response = client.post("/model/forward", json={"text": text_100})
        data = response.json()
        assert data.get("truncated") is False or data.get("truncated") is None
        
        # 101 tokens - should truncate (use single char words to stay under 500 char limit)
        text_101 = " ".join(["a"] * 101)
        response = client.post("/model/forward", json={"text": text_101})
        data = response.json()
        assert data["truncated"] is True
    
    def test_truncation_warning_message(self, client, large_model_request):
        """Test that truncation includes appropriate warning message."""
        response = client.post("/model/forward", json=large_model_request)
        data = response.json()
        
        assert data["warnings"] is not None
        assert len(data["warnings"]) > 0
        
        # Check that warning mentions truncation
        warning_text = " ".join(data["warnings"]).lower()
        assert "truncat" in warning_text or "100" in warning_text
    
    def test_tensor_preservation_before_truncation(self):
        """Test that tensor data is preserved correctly before truncation point."""
        max_elements = 100
        tensor = np.arange(200).reshape(20, 10)
        
        truncated, was_truncated = truncate_tensor_data(tensor, max_elements)
        
        assert was_truncated
        # Check that preserved portion matches original
        expected_dim = int(np.sqrt(max_elements))
        np.testing.assert_array_equal(
            truncated,
            tensor[:expected_dim, :expected_dim]
        )
    
    def test_multiple_truncations_consistent(self):
        """Test that multiple truncations of same tensor are consistent."""
        max_elements = 500
        tensor = np.random.randn(100, 100)
        
        truncated1, _ = truncate_tensor_data(tensor.copy(), max_elements)
        truncated2, _ = truncate_tensor_data(tensor.copy(), max_elements)
        
        np.testing.assert_array_equal(truncated1, truncated2)
    
    def test_edge_case_exact_max_elements(self):
        """Test tensor with exactly max_elements is not truncated."""
        max_elements = 100
        tensor = np.random.randn(10, 10)
        
        truncated, was_truncated = truncate_tensor_data(tensor, max_elements)
        
        assert not was_truncated
        assert truncated.size == 100
        np.testing.assert_array_equal(truncated, tensor)
    
    def test_attention_data_structure_with_truncation(self, client):
        """Test that attention data structure is correct when truncated."""
        # Test with different input sizes (use single char to avoid exceeding 500 char limit)
        for num_words in [5, 20, 105]:
            text = " ".join(["a"] * num_words)
            response = client.post("/model/forward", json={"text": text})
            data = response.json()
            
            for step in data["steps"]:
                layer_data = step["layerData"]
                if layer_data.get("layerType") == "attention":
                    if layer_data.get("truncated"):
                        # Should have None attention data when truncated
                        assert layer_data.get("attentionData") is None
                    else:
                        # Should have attention data when not truncated
                        assert layer_data.get("attentionData") is not None
    
    def test_moe_data_structure_with_truncation(self, client):
        """Test that MoE data structure is correct when truncated."""
        # Test with different input sizes (use single char to avoid exceeding 500 char limit)
        for num_words in [5, 20, 105]:
            text = " ".join(["a"] * num_words)
            response = client.post("/model/forward", json={"text": text})
            data = response.json()
            
            for step in data["steps"]:
                layer_data = step["layerData"]
                if layer_data.get("layerType") == "moe":
                    if layer_data.get("truncated"):
                        # Should have None MoE data when truncated
                        assert layer_data.get("moeData") is None
                    else:
                        # Should have MoE data when not truncated
                        assert layer_data.get("moeData") is not None
