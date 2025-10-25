"""
Tests for forward tracing endpoint (/model/forward/traced).
Verifies the new API with ForwardStep, LayerState, and ModelRunResponse schemas.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.mark.integration
class TestForwardTracingEndpoint:
    """Tests for the new /model/forward/traced endpoint."""
    
    def test_traced_endpoint_exists(self, client):
        """Test that /model/forward/traced endpoint is accessible."""
        response = client.post("/model/forward/traced", json={"text": "Hello"})
        assert response.status_code == 200
    
    def test_traced_returns_model_run_response(self, client):
        """Test that traced endpoint returns ModelRunResponse structure."""
        response = client.post("/model/forward/traced", json={"text": "Hello world"})
        assert response.status_code == 200
        data = response.json()
        
        # Check top-level keys
        required_keys = [
            "success", "inputText", "tokens", "tokenCount",
            "steps", "finalLogits", "outputProbabilities", "metadata"
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
        
        assert data["success"] is True
    
    def test_traced_returns_forward_steps(self, client):
        """Test that steps contain ForwardStep structure."""
        response = client.post("/model/forward/traced", json={"text": "Test"})
        data = response.json()
        
        assert "steps" in data
        assert len(data["steps"]) > 0
        
        # Check first step structure
        step = data["steps"][0]
        assert "stepIndex" in step
        assert "layerState" in step
        assert "description" in step
        assert "timingMs" in step
    
    def test_traced_layer_state_structure(self, client):
        """Test that layerState has complete structure with pre/post activations."""
        response = client.post("/model/forward/traced", json={"text": "Hi"})
        data = response.json()
        
        for step in data["steps"]:
            layer_state = step["layerState"]
            
            # Required fields
            assert "layerId" in layer_state
            assert "layerName" in layer_state
            assert "layerType" in layer_state
            assert "inputShape" in layer_state
            assert "outputShape" in layer_state
            assert "metadata" in layer_state
            
            # At least one of pre/post activations should exist
            has_activations = (
                layer_state.get("preActivations") is not None or
                layer_state.get("postActivations") is not None
            )
            assert has_activations, f"Layer {layer_state['layerId']} has no activations"
    
    def test_traced_tensor_metadata(self, client):
        """Test that TracedTensor includes metadata with statistics."""
        response = client.post("/model/forward/traced", json={"text": "Test"})
        data = response.json()
        
        # Check post activations of first step
        first_step = data["steps"][0]
        post_activations = first_step["layerState"].get("postActivations")
        
        if post_activations:
            assert "name" in post_activations
            assert "data" in post_activations
            assert "metadata" in post_activations
            
            metadata = post_activations["metadata"]
            assert "shape" in metadata
            assert "dtype" in metadata
            assert "truncated" in metadata
            assert "minVal" in metadata
            assert "maxVal" in metadata
            assert "meanVal" in metadata
            assert "stdVal" in metadata
    
    def test_traced_attention_state(self, client):
        """Test that attention layers have AttentionState."""
        response = client.post("/model/forward/traced", json={"text": "Hello"})
        data = response.json()
        
        # Find attention layer
        attention_steps = [
            step for step in data["steps"]
            if step["layerState"]["layerType"] == "attention"
        ]
        
        assert len(attention_steps) > 0, "No attention layers found"
        
        for step in attention_steps:
            attention_state = step["layerState"].get("attentionState")
            assert attention_state is not None, "Attention layer missing attentionState"
            
            # Check attention state structure
            assert "queryMatrix" in attention_state
            assert "keyMatrix" in attention_state
            assert "valueMatrix" in attention_state
            assert "attentionScores" in attention_state
            assert "numHeads" in attention_state
            assert "headDim" in attention_state
    
    def test_traced_moe_state(self, client):
        """Test that MoE layers have MoEState."""
        response = client.post("/model/forward/traced", json={"text": "Test"})
        data = response.json()
        
        # Find MoE layer
        moe_steps = [
            step for step in data["steps"]
            if step["layerState"]["layerType"] == "moe"
        ]
        
        assert len(moe_steps) > 0, "No MoE layers found"
        
        for step in moe_steps:
            moe_state = step["layerState"].get("moeState")
            assert moe_state is not None, "MoE layer missing moeState"
            
            # Check MoE state structure
            assert "gatingWeights" in moe_state
            assert "selectedExperts" in moe_state
            assert "expertActivations" in moe_state
            assert "numExperts" in moe_state
            assert "topK" in moe_state
    
    def test_traced_model_metadata(self, client):
        """Test that response includes model metadata."""
        response = client.post("/model/forward/traced", json={"text": "Test"})
        data = response.json()
        
        metadata = data["metadata"]
        assert "vocabSize" in metadata
        assert "embeddingDim" in metadata
        assert "hiddenDim" in metadata
        assert "numLayers" in metadata
        assert "tokenizerType" in metadata
        assert "totalTimeMs" in metadata
        
        # Verify types
        assert isinstance(metadata["vocabSize"], int)
        assert isinstance(metadata["totalTimeMs"], (int, float))
    
    def test_traced_tokenizer_bpe(self, client):
        """Test BPE tokenizer produces correct tokens."""
        response = client.post(
            "/model/forward/traced?tokenizer_type=bpe",
            json={"text": "Hello world"}
        )
        data = response.json()
        
        assert data["metadata"]["tokenizerType"] == "bpe"
        assert data["tokenCount"] == 2
        assert len(data["tokens"]) == 2
        
        # Check token structure
        for token in data["tokens"]:
            assert "text" in token
            assert "id" in token
            assert "startPos" in token
            assert "endPos" in token
    
    def test_traced_tokenizer_char(self, client):
        """Test character tokenizer produces correct tokens."""
        response = client.post(
            "/model/forward/traced?tokenizer_type=char",
            json={"text": "Hi"}
        )
        data = response.json()
        
        assert data["metadata"]["tokenizerType"] == "char"
        assert data["tokenCount"] == 2
        assert len(data["tokens"]) == 2
        assert data["tokens"][0]["text"] == "H"
        assert data["tokens"][1]["text"] == "i"
    
    def test_traced_step_index_parameter(self, client):
        """Test step_index query parameter returns only specified step."""
        # First, get all steps
        response_all = client.post(
            "/model/forward/traced",
            json={"text": "Test"}
        )
        all_steps = response_all.json()["steps"]
        num_steps = len(all_steps)
        
        # Request specific step
        response_single = client.post(
            "/model/forward/traced?step_index=1",
            json={"text": "Test"}
        )
        data = response_single.json()
        
        assert len(data["steps"]) == 1
        assert data["steps"][0]["stepIndex"] == 1
        assert data["steps"][0]["layerState"]["layerId"] == 1
    
    def test_traced_step_index_invalid(self, client):
        """Test invalid step_index returns error."""
        response = client.post(
            "/model/forward/traced?step_index=999",
            json={"text": "Test"}
        )
        assert response.status_code == 400
        assert "Invalid step_index" in response.json()["detail"]
    
    def test_traced_timing_metadata(self, client):
        """Test that timing information is included."""
        response = client.post("/model/forward/traced", json={"text": "Test"})
        data = response.json()
        
        # Check overall timing
        assert "totalTimeMs" in data["metadata"]
        assert data["metadata"]["totalTimeMs"] > 0
        
        # Check per-step timing
        for step in data["steps"]:
            assert "timingMs" in step
            if step["timingMs"] is not None:
                assert step["timingMs"] > 0
    
    def test_traced_final_logits(self, client):
        """Test that finalLogits tensor is included."""
        response = client.post("/model/forward/traced", json={"text": "Test"})
        data = response.json()
        
        assert "finalLogits" in data
        final_logits = data["finalLogits"]
        
        assert "name" in final_logits
        assert "data" in final_logits
        assert "metadata" in final_logits
        assert isinstance(final_logits["data"], list)
    
    def test_traced_output_probabilities(self, client):
        """Test that output probabilities are computed correctly."""
        response = client.post("/model/forward/traced", json={"text": "Test"})
        data = response.json()
        
        probs = data["outputProbabilities"]
        assert len(probs) > 0
        assert len(probs) <= 5  # Top-5
        
        # Check probability structure
        for prob in probs:
            assert "token" in prob
            assert "probability" in prob
            assert 0 <= prob["probability"] <= 1
    
    def test_traced_truncation_warning(self, client):
        """Test that warning is issued for large token counts."""
        # Create text with >16 tokens
        long_text = " ".join([f"word{i}" for i in range(20)])
        response = client.post(
            "/model/forward/traced?tokenizer_type=bpe",
            json={"text": long_text}
        )
        data = response.json()
        
        assert data["tokenCount"] > 16
        assert data["warnings"] is not None
        assert len(data["warnings"]) > 0
        assert "exceeds recommended limit" in data["warnings"][0]
    
    def test_traced_payload_size_limit(self, client):
        """Test that payload size stays under 1MB for <=16 tokens."""
        # Test with 16 tokens
        text = " ".join([f"token{i}" for i in range(16)])
        response = client.post(
            "/model/forward/traced?tokenizer_type=bpe",
            json={"text": text}
        )
        
        assert response.status_code == 200
        
        # Check response size
        content_length = len(response.content)
        assert content_length < 1024 * 1024, f"Payload size {content_length} exceeds 1MB"
    
    def test_traced_empty_text_validation(self, client):
        """Test that empty text is rejected."""
        response = client.post("/model/forward/traced", json={"text": ""})
        assert response.status_code == 422
    
    def test_traced_text_too_long_validation(self, client):
        """Test that text exceeding max length is rejected."""
        long_text = "a" * 501
        response = client.post("/model/forward/traced", json={"text": long_text})
        assert response.status_code == 422
    
    def test_traced_invalid_tokenizer_type(self, client):
        """Test that invalid tokenizer type returns error."""
        response = client.post(
            "/model/forward/traced?tokenizer_type=invalid",
            json={"text": "Test"}
        )
        assert response.status_code == 400
        assert "Invalid tokenizer type" in response.json()["detail"]
    
    def test_traced_consistency_across_calls(self, client):
        """Test that multiple calls with same input produce consistent structure."""
        text = "Hello world"
        
        response1 = client.post("/model/forward/traced", json={"text": text})
        response2 = client.post("/model/forward/traced", json={"text": text})
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Check structural consistency
        assert data1["tokenCount"] == data2["tokenCount"]
        assert len(data1["steps"]) == len(data2["steps"])
        assert data1["metadata"]["numLayers"] == data2["metadata"]["numLayers"]
        
        # Check layer structure consistency
        for step1, step2 in zip(data1["steps"], data2["steps"]):
            assert step1["layerState"]["layerType"] == step2["layerState"]["layerType"]
            assert step1["layerState"]["inputShape"] == step2["layerState"]["inputShape"]
            assert step1["layerState"]["outputShape"] == step2["layerState"]["outputShape"]


@pytest.mark.unit
class TestTokenizers:
    """Tests for tokenizer implementations."""
    
    def test_char_tokenizer(self):
        """Test character tokenizer."""
        from app.tokenizer import CharTokenizer
        
        tokenizer = CharTokenizer()
        token_ids, token_info = tokenizer.encode("Hi")
        
        assert len(token_ids) == 2
        assert len(token_info) == 2
        assert token_info[0].text == "H"
        assert token_info[1].text == "i"
        
        # Test decode
        decoded = tokenizer.decode(token_ids)
        assert decoded == "Hi"
    
    def test_bpe_tokenizer(self):
        """Test BPE tokenizer."""
        from app.tokenizer import SimpleBPETokenizer
        
        tokenizer = SimpleBPETokenizer()
        token_ids, token_info = tokenizer.encode("Hello world")
        
        assert len(token_ids) == 2
        assert token_info[0].text == "Hello"
        assert token_info[1].text == "world"
    
    def test_tokenizer_factory(self):
        """Test tokenizer factory function."""
        from app.tokenizer import get_tokenizer
        
        char_tok = get_tokenizer("char")
        assert char_tok is not None
        
        bpe_tok = get_tokenizer("bpe")
        assert bpe_tok is not None
        
        # Invalid type should raise error
        with pytest.raises(ValueError):
            get_tokenizer("invalid")


@pytest.mark.unit
class TestSimpleTransformerModel:
    """Tests for SimpleTransformerModel."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        from app.model import SimpleTransformerModel
        
        model = SimpleTransformerModel(
            vocab_size=256,
            embedding_dim=64,
            hidden_dim=128
        )
        
        assert model.vocab_size == 256
        assert model.embedding_dim == 64
        assert model.hidden_dim == 128
    
    def test_model_forward_pass(self):
        """Test model forward pass produces traces."""
        from app.model import SimpleTransformerModel
        
        model = SimpleTransformerModel(vocab_size=100)
        token_ids = [10, 20, 30]
        
        traces = model.forward(token_ids)
        
        assert len(traces) > 0
        assert all(hasattr(trace, 'layer_id') for trace in traces)
        assert all(hasattr(trace, 'layer_name') for trace in traces)
        assert all(hasattr(trace, 'layer_type') for trace in traces)
    
    def test_model_tensor_tracing(self):
        """Test that tensors are traced with statistics."""
        from app.model import SimpleTransformerModel
        
        model = SimpleTransformerModel(vocab_size=50)
        traces = model.forward([5, 10])
        
        # Check first trace has post_activations
        first_trace = traces[0]
        assert first_trace.post_activations is not None
        
        tensor_trace = first_trace.post_activations
        assert hasattr(tensor_trace, 'min_val')
        assert hasattr(tensor_trace, 'max_val')
        assert hasattr(tensor_trace, 'mean_val')
        assert hasattr(tensor_trace, 'std_val')
        assert hasattr(tensor_trace, 'truncated')
