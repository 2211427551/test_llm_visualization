import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestHealthAPI:
    """健康检查 API 测试类"""
    
    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get("/api/v1/health/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "services" in data
        assert data["services"]["app"] == "healthy"
    
    def test_app_info(self, client):
        """测试应用信息接口"""
        response = client.get("/api/v1/health/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "debug" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self, client):
        """测试根路径接口"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "docs" in data
        assert "health" in data


class TestApplicationStartup:
    """应用启动测试类"""
    
    def test_app_creation(self):
        """测试应用创建"""
        from app.main import create_app
        
        app = create_app()
        
        assert app.title == "FastAPI 后端应用"
        assert app.version == "0.1.0"
        assert app.debug is True
    
    def test_config_loading(self):
        """测试配置加载"""
        from app.core.config import settings
        
        assert settings.app_name == "FastAPI 后端应用"
        assert settings.app_version == "0.1.0"
        assert settings.debug is True
        assert settings.api_prefix == "/api/v1"