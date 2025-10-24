import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_model_forward_success():
    response = client.post(
        "/model/forward",
        json={"text": "Hello world"}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["inputText"] == "Hello world"
    assert data["tokenCount"] == 2
    assert len(data["tokens"]) == 2
    assert len(data["steps"]) == 6
    assert len(data["outputProbabilities"]) > 0


def test_model_forward_empty_text():
    response = client.post(
        "/model/forward",
        json={"text": ""}
    )
    assert response.status_code == 422


def test_model_forward_too_long():
    long_text = "a" * 501
    response = client.post(
        "/model/forward",
        json={"text": long_text}
    )
    assert response.status_code == 422


def test_model_forward_truncation_warning():
    long_text = " ".join(["word"] * 101)
    response = client.post(
        "/model/forward",
        json={"text": long_text}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["truncated"] is True
    assert data["warnings"] is not None
    assert len(data["warnings"]) > 0


def test_model_forward_layer_structure():
    response = client.post(
        "/model/forward",
        json={"text": "Test input"}
    )
    data = response.json()
    
    for step in data["steps"]:
        assert "stepIndex" in step
        assert "layerData" in step
        assert "description" in step
        
        layer_data = step["layerData"]
        assert "layerId" in layer_data
        assert "layerName" in layer_data
        assert "inputShape" in layer_data
        assert "outputShape" in layer_data
