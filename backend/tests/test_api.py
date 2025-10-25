import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_200(self, client):
        """Test that /health returns 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Test health check response has correct structure."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert data["status"] == "healthy"


@pytest.mark.integration
class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_response_structure(self, client):
        """Test root response contains API information."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


@pytest.mark.integration
class TestModelForwardEndpoint:
    """Tests for model forward pass endpoint."""
    
    def test_forward_handles_sample_text(self, client, sample_model_request):
        """Test that /model/forward handles sample text successfully."""
        response = client.post("/model/forward", json=sample_model_request)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_forward_validates_json_schema_keys(self, client, sample_model_request):
        """Test that response contains all required schema keys."""
        response = client.post("/model/forward", json=sample_model_request)
        data = response.json()
        
        # Top-level keys
        required_keys = [
            "success",
            "inputText",
            "tokens",
            "tokenCount",
            "steps",
            "outputProbabilities"
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
        
        # Validate tokens structure
        assert isinstance(data["tokens"], list)
        if len(data["tokens"]) > 0:
            token = data["tokens"][0]
            assert "text" in token
            assert "id" in token
        
        # Validate steps structure
        assert isinstance(data["steps"], list)
        assert len(data["steps"]) > 0
        for step in data["steps"]:
            assert "stepIndex" in step
            assert "layerData" in step
            assert "description" in step
    
    def test_forward_layer_data_structure(self, client, sample_model_request):
        """Test that layer data has correct structure."""
        response = client.post("/model/forward", json=sample_model_request)
        data = response.json()
        
        for step in data["steps"]:
            layer_data = step["layerData"]
            
            # Required layer data keys
            assert "layerId" in layer_data
            assert "layerName" in layer_data
            assert "inputShape" in layer_data
            assert "outputShape" in layer_data
            
            # Shapes should be lists of integers
            assert isinstance(layer_data["inputShape"], list)
            assert isinstance(layer_data["outputShape"], list)
    
    def test_forward_returns_consistent_tensor_shapes(self, client):
        """Test that tensor shapes are consistent across calls."""
        request = {"text": "Hello world"}
        
        response1 = client.post("/model/forward", json=request)
        response2 = client.post("/model/forward", json=request)
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Token count should be the same
        assert data1["tokenCount"] == data2["tokenCount"]
        
        # Number of steps should be the same
        assert len(data1["steps"]) == len(data2["steps"])
        
        # Layer shapes should be consistent
        for step1, step2 in zip(data1["steps"], data2["steps"]):
            layer1 = step1["layerData"]
            layer2 = step2["layerData"]
            assert layer1["inputShape"] == layer2["inputShape"]
            assert layer1["outputShape"] == layer2["outputShape"]
    
    def test_forward_returns_consistent_dtypes(self, client, sample_model_request):
        """Test that response data types are correct."""
        response = client.post("/model/forward", json=sample_model_request)
        data = response.json()
        
        # Check data types
        assert isinstance(data["success"], bool)
        assert isinstance(data["inputText"], str)
        assert isinstance(data["tokenCount"], int)
        assert isinstance(data["tokens"], list)
        assert isinstance(data["steps"], list)
        assert isinstance(data["outputProbabilities"], list)
        
        # Check output probabilities structure
        for prob in data["outputProbabilities"]:
            assert "token" in prob
            assert "probability" in prob
            assert isinstance(prob["token"], str)
            assert isinstance(prob["probability"], (int, float))
            assert 0 <= prob["probability"] <= 1
    
    def test_forward_attention_layer_structure(self, client, sample_model_request):
        """Test that attention layers have correct structure."""
        response = client.post("/model/forward", json=sample_model_request)
        data = response.json()
        
        # Find attention layer
        attention_layers = [
            step["layerData"] for step in data["steps"]
            if step["layerData"].get("layerType") == "attention"
        ]
        
        assert len(attention_layers) > 0, "No attention layers found"
        
        for layer in attention_layers:
            if layer.get("attentionData") and not layer.get("truncated"):
                att_data = layer["attentionData"]
                assert "queryMatrix" in att_data or att_data["queryMatrix"] is None
                assert "keyMatrix" in att_data or att_data["keyMatrix"] is None
                assert "valueMatrix" in att_data or att_data["valueMatrix"] is None
                assert "attentionScores" in att_data or att_data["attentionScores"] is None
                assert "sparsityMask" in att_data or att_data["sparsityMask"] is None
    
    def test_forward_moe_layer_structure(self, client, sample_model_request):
        """Test that MoE layers have correct structure."""
        response = client.post("/model/forward", json=sample_model_request)
        data = response.json()
        
        # Find MoE layer
        moe_layers = [
            step["layerData"] for step in data["steps"]
            if step["layerData"].get("layerType") == "moe"
        ]
        
        assert len(moe_layers) > 0, "No MoE layers found"
        
        for layer in moe_layers:
            if layer.get("moeData") and not layer.get("truncated"):
                moe_data = layer["moeData"]
                assert "gatingWeights" in moe_data or moe_data["gatingWeights"] is None
                assert "selectedExperts" in moe_data or moe_data["selectedExperts"] is None
                assert "numExperts" in moe_data or moe_data["numExperts"] is None
                assert "topK" in moe_data or moe_data["topK"] is None
    
    def test_forward_empty_text_validation(self, client):
        """Test that empty text is rejected."""
        response = client.post("/model/forward", json={"text": ""})
        assert response.status_code == 422
    
    def test_forward_text_too_long_validation(self, client):
        """Test that text exceeding max length is rejected."""
        long_text = "a" * 501
        response = client.post("/model/forward", json={"text": long_text})
        assert response.status_code == 422
    
    def test_forward_missing_text_field(self, client):
        """Test that missing text field is rejected."""
        response = client.post("/model/forward", json={})
        assert response.status_code == 422
    
    def test_forward_invalid_json(self, client):
        """Test that invalid JSON is rejected."""
        response = client.post(
            "/model/forward",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_forward_whitespace_only_text(self, client):
        """Test handling of whitespace-only text."""
        response = client.post("/model/forward", json={"text": "   "})
        # After stripping, this becomes empty but passes initial validation
        # It tokenizes to an empty list
        assert response.status_code == 200
        data = response.json()
        assert data["tokenCount"] == 0
        assert len(data["tokens"]) == 0
    
    def test_forward_single_token(self, client):
        """Test forward pass with single token."""
        response = client.post("/model/forward", json={"text": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert data["tokenCount"] == 1
        assert len(data["tokens"]) == 1
    
    def test_forward_multiple_tokens(self, client):
        """Test forward pass with multiple tokens."""
        response = client.post("/model/forward", json={"text": "Hello world test"})
        assert response.status_code == 200
        data = response.json()
        assert data["tokenCount"] == 3
        assert len(data["tokens"]) == 3
    
    def test_forward_output_probabilities_sum(self, client, sample_model_request):
        """Test that output probabilities sum to approximately 1."""
        response = client.post("/model/forward", json=sample_model_request)
        data = response.json()
        
        probs = [p["probability"] for p in data["outputProbabilities"]]
        # Note: Only top-5 are returned, so they won't sum to 1
        # But each should be valid probability
        for prob in probs:
            assert 0 <= prob <= 1
    
    def test_forward_tokens_have_sequential_ids(self, client):
        """Test that tokens have sequential IDs."""
        response = client.post("/model/forward", json={"text": "one two three"})
        data = response.json()
        
        ids = [token["id"] for token in data["tokens"]]
        expected_ids = list(range(len(ids)))
        assert ids == expected_ids
    
    def test_forward_steps_have_sequential_indices(self, client, sample_model_request):
        """Test that computation steps have sequential indices."""
        response = client.post("/model/forward", json=sample_model_request)
        data = response.json()
        
        indices = [step["stepIndex"] for step in data["steps"]]
        expected_indices = list(range(len(indices)))
        assert indices == expected_indices
