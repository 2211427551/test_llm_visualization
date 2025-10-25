# Backend API

FastAPI backend for the Model Execution Visualizer.

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Run the development server:
```bash
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /model/forward` - Run model forward pass
- `GET /docs` - Interactive API documentation (Swagger UI)

## Testing

The backend includes comprehensive automated tests covering API endpoints, model utilities, and regression scenarios.

### Running Tests

Run all tests:
```bash
poetry run pytest
```

Run tests with verbose output:
```bash
poetry run pytest -v
```

Run specific test categories:
```bash
# Unit tests only
poetry run pytest -m unit

# Integration tests only
poetry run pytest -m integration

# Regression tests only
poetry run pytest -m regression
```

Run tests from a specific file:
```bash
poetry run pytest tests/test_api.py
poetry run pytest tests/test_model_utils.py
poetry run pytest tests/test_serialization.py
```

Run tests with coverage report:
```bash
poetry run pytest --cov=app --cov-report=html
```

### Test Coverage

The test suite covers:

- **API Endpoints** (`tests/test_api.py`)
  - Health check endpoint validation
  - Model forward pass with various input sizes
  - JSON schema validation
  - Tensor shape and dtype consistency
  - Error handling and validation

- **Model Utilities** (`tests/test_model_utils.py`)
  - Sparse attention mask construction
  - MoE gating mechanism (top-k selection, weight normalization)
  - Tensor truncation behavior
  - Attention sparsity validation

- **Serialization & Truncation** (`tests/test_serialization.py`)
  - Regression tests for truncation behavior
  - Max element limits enforcement
  - Truncation threshold validation
  - Warning message generation

### CI Integration

For continuous integration, add the following command to your CI pipeline:

```bash
cd backend && poetry install && poetry run pytest -v --tb=short
```

This ensures all tests pass before merging changes.
