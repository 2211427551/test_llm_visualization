import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture
def seeded_rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "Hello world this is a test"


@pytest.fixture
def sample_token_embeddings():
    """Sample token embeddings for testing."""
    np.random.seed(42)
    num_tokens = 10
    embedding_dim = 64
    return np.random.randn(num_tokens, embedding_dim)


@pytest.fixture
def sample_attention_scores():
    """Sample attention scores matrix."""
    np.random.seed(42)
    num_tokens = 8
    raw_scores = np.random.randn(num_tokens, num_tokens)
    attention_scores = np.exp(raw_scores) / np.exp(raw_scores).sum(axis=1, keepdims=True)
    return attention_scores


@pytest.fixture
def sample_model_request():
    """Sample model forward request data."""
    return {"text": "Hello world"}


@pytest.fixture
def large_model_request():
    """Large model request that triggers truncation (>100 tokens, <500 chars)."""
    return {"text": " ".join(["a"] * 101)}
