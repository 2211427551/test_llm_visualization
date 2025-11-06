import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_inference_service


@pytest.fixture(autouse=True)
def reset_model_service():
    """每个用例执行前重置模型状态，避免状态污染。"""
    model_inference_service.reset()
    yield
    model_inference_service.reset()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_initialize_returns_config(client: TestClient):
    response = client.get("/api/v1/initialize")
    assert response.status_code == 200
    payload = response.json()

    assert payload["success"] is True
    assert payload["message"] == "模型初始化成功"
    config = payload["config"]
    assert config["modelName"] == "Transformer-MoE"
    assert "initializedAt" in config


def test_forward_requires_initialization(client: TestClient):
    response = client.post("/api/v1/forward", json={"text": "你好", "capture_data": True})
    assert response.status_code == 400
    assert "模型未初始化" in response.json()["detail"]


def test_forward_returns_visualization(client: TestClient):
    init_resp = client.get("/api/v1/initialize")
    assert init_resp.status_code == 200

    forward_resp = client.post(
        "/api/v1/forward",
        json={"text": "你好，Transformer", "capture_data": True},
    )
    assert forward_resp.status_code == 200
    payload = forward_resp.json()

    assert payload["success"] is True
    assert payload["sequence_length"] > 0
    assert payload["logits_shape"] == [1, payload["sequence_length"], init_resp.json()["config"]["vocabSize"]]

    captured = payload["captured_data"]
    assert captured is not None
    assert captured["steps"], "推理结果应包含步骤可视化数据"
    assert captured["runtime"]["sequenceLength"] == payload["sequence_length"]


def test_forward_validates_empty_text(client: TestClient):
    client.get("/api/v1/initialize")

    response = client.post(
        "/api/v1/forward",
        json={"text": "   ", "capture_data": True},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "输入文本不能为空"
