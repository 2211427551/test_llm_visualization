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

Run tests with:
```bash
poetry run pytest
```
