# Data Visualization Platform

A full-stack application for visualizing complex data structures with interactive macro and micro views. The platform provides a robust API for data ingestion and a responsive frontend for exploring datasets at multiple levels of detail.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Environment Variables](#environment-variables)
- [API Documentation](#api-documentation)
  - [Endpoints](#endpoints)
  - [Request/Response Examples](#requestresponse-examples)
  - [Payload Size Considerations](#payload-size-considerations)
- [Visualization Guide](#visualization-guide)
  - [Macro View](#macro-view)
  - [Micro View](#micro-view)
  - [Step Controls](#step-controls)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This application enables users to upload, process, and visualize large datasets through an intuitive interface. The platform is designed to handle complex data structures and provide insights through multiple visualization layers, from high-level overviews to detailed data point exploration.

## Architecture

The project follows a modern full-stack architecture:

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **Server**: Uvicorn ASGI server
- **Dependency Management**: Poetry
- **Key Components**:
  - RESTful API for data ingestion and retrieval
  - Data processing and transformation pipeline
  - Validation and error handling middleware
  - Async request handling for optimal performance

### Frontend
- **Package Manager**: npm
- **Key Components**:
  - Interactive visualization components
  - State management for data exploration
  - Responsive UI with macro/micro view switching
  - Step-by-step navigation controls

```
┌─────────────────────────────────────────┐
│           Frontend (React/Vue)          │
│  ┌────────────┐      ┌────────────────┐ │
│  │ Macro View │      │   Micro View   │ │
│  └────────────┘      └────────────────┘ │
│         │                    │          │
│         └──────────┬─────────┘          │
│                    │                    │
└────────────────────┼────────────────────┘
                     │ HTTP/REST
┌────────────────────┼────────────────────┐
│                    │                    │
│         ┌──────────▼─────────┐          │
│         │   API Routes       │          │
│         ├────────────────────┤          │
│         │ Data Processing    │          │
│         ├────────────────────┤          │
│         │   Data Storage     │          │
│         └────────────────────┘          │
│         Backend (FastAPI)              │
└─────────────────────────────────────────┘
```

## Features

- **Multi-level Visualization**: Switch between macro (overview) and micro (detailed) views
- **Interactive Navigation**: Step-by-step controls for exploring data sequences
- **Real-time Data Processing**: Async API handles large payloads efficiently
- **Responsive Design**: Optimized for desktop and tablet devices
- **Data Validation**: Comprehensive input validation and error handling
- **RESTful API**: Clean, documented API endpoints for integration

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python**: 3.9 or higher
- **Poetry**: 1.5+ (Python dependency management)
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher
- **Git**: For version control

## Setup Instructions

### Backend Setup

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install
   ```

3. **Activate the Poetry virtual environment**:
   ```bash
   poetry shell
   ```

4. **Set up environment variables** (see [Environment Variables](#environment-variables) section).

5. **Run the development server**:
   ```bash
   poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`
   
   Interactive API documentation (Swagger UI) will be available at `http://localhost:8000/docs`

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables** (see [Environment Variables](#environment-variables) section).

4. **Run the development server**:
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:3000` (or the port specified in your configuration)

### Environment Variables

#### Backend (`.env` in `backend/`)

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Database (if applicable)
DATABASE_URL=sqlite:///./data.db

# API Settings
MAX_PAYLOAD_SIZE=10485760  # 10MB in bytes
API_KEY_REQUIRED=false

# Logging
LOG_LEVEL=INFO
```

#### Frontend (`.env` in `frontend/`)

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=30000

# Feature Flags
VITE_ENABLE_MICRO_VIEW=true
VITE_ENABLE_STEP_CONTROLS=true

# Visualization Settings
VITE_MAX_DATA_POINTS=10000
VITE_DEFAULT_VIEW=macro
```

## API Documentation

### Endpoints

#### Health Check
```
GET /health
```
Returns the health status of the API.

#### Data Upload
```
POST /api/data/upload
```
Upload dataset for visualization.

#### Get Data
```
GET /api/data/{data_id}
```
Retrieve processed data by ID.

#### List Datasets
```
GET /api/data/list
```
List all available datasets.

### Request/Response Examples

#### Upload Data

**Request**:
```bash
curl -X POST "http://localhost:8000/api/data/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sales Data Q4 2024",
    "data": [
      {"timestamp": "2024-10-01T00:00:00Z", "value": 150.5, "category": "A"},
      {"timestamp": "2024-10-02T00:00:00Z", "value": 200.3, "category": "B"},
      {"timestamp": "2024-10-03T00:00:00Z", "value": 175.8, "category": "A"}
    ],
    "metadata": {
      "source": "sales_system",
      "unit": "USD"
    }
  }'
```

**Response** (201 Created):
```json
{
  "id": "d4f8a2b1-3c5e-4d7a-9f2b-8e6c1a3d5f7e",
  "name": "Sales Data Q4 2024",
  "status": "processed",
  "data_points": 3,
  "created_at": "2024-10-24T15:30:00Z",
  "summary": {
    "min_value": 150.5,
    "max_value": 200.3,
    "avg_value": 175.53
  }
}
```

#### Retrieve Data

**Request**:
```bash
curl -X GET "http://localhost:8000/api/data/d4f8a2b1-3c5e-4d7a-9f2b-8e6c1a3d5f7e"
```

**Response** (200 OK):
```json
{
  "id": "d4f8a2b1-3c5e-4d7a-9f2b-8e6c1a3d5f7e",
  "name": "Sales Data Q4 2024",
  "data": [
    {"timestamp": "2024-10-01T00:00:00Z", "value": 150.5, "category": "A"},
    {"timestamp": "2024-10-02T00:00:00Z", "value": 200.3, "category": "B"},
    {"timestamp": "2024-10-03T00:00:00Z", "value": 175.8, "category": "A"}
  ],
  "metadata": {
    "source": "sales_system",
    "unit": "USD"
  },
  "visualization_config": {
    "macro_view": {
      "type": "line_chart",
      "aggregation": "daily"
    },
    "micro_view": {
      "type": "scatter_plot",
      "detail_level": "full"
    }
  }
}
```

#### Error Response

**Response** (400 Bad Request):
```json
{
  "error": "validation_error",
  "message": "Invalid data format",
  "details": {
    "field": "data[0].timestamp",
    "issue": "Invalid ISO 8601 format"
  }
}
```

### Payload Size Considerations

The API is optimized to handle varying payload sizes:

- **Small Payloads** (<100KB): Processed synchronously with immediate response
- **Medium Payloads** (100KB - 1MB): Processed asynchronously with status polling
- **Large Payloads** (1MB - 10MB): Chunked processing with progress updates
- **Maximum Size**: 10MB (configurable via `MAX_PAYLOAD_SIZE` environment variable)

**Best Practices**:
1. **Compression**: Use gzip compression for requests over 100KB
2. **Pagination**: For large datasets, prefer multiple smaller requests
3. **Streaming**: Use streaming endpoints for continuous data feeds
4. **Batch Operations**: Group multiple small operations into batch requests

**Example with compression**:
```bash
curl -X POST "http://localhost:8000/api/data/upload" \
  -H "Content-Type: application/json" \
  -H "Content-Encoding: gzip" \
  --data-binary @compressed_data.json.gz
```

## Visualization Guide

The platform provides two complementary visualization modes to explore your data effectively.

### Macro View

The **Macro View** provides a high-level overview of your entire dataset, perfect for identifying trends, patterns, and outliers.

**Features**:
- Aggregated data representation
- Timeline or summary charts
- Quick filtering and sorting
- Export capabilities

**Usage**:
1. Select a dataset from the dashboard
2. The macro view loads automatically
3. Use the toolbar to adjust aggregation levels (hourly, daily, weekly)
4. Click on any data point to drill down to micro view

**Screenshot Placeholder**:
```
[TODO: Add screenshot of macro view showing line chart with time series data]
```

### Micro View

The **Micro View** allows detailed inspection of individual data points and their relationships.

**Features**:
- Detailed data point information
- Zoom and pan capabilities
- Tooltip with full metadata
- Individual point highlighting

**Usage**:
1. Click on a data point or region in macro view
2. Micro view opens with focused data
3. Hover over points for detailed information
4. Use zoom controls to adjust detail level

**Screenshot Placeholder**:
```
[TODO: Add screenshot of micro view showing detailed scatter plot]
```

### Step Controls

Navigate through your data sequentially using step controls, ideal for time-series or ordered datasets.

**Controls**:
- **First** (⏮): Jump to the first data point
- **Previous** (◀): Move to the previous step
- **Play/Pause** (▶/⏸): Auto-advance through steps
- **Next** (▶): Move to the next step
- **Last** (⏭): Jump to the last data point

**Keyboard Shortcuts**:
- `Space`: Play/Pause
- `←`: Previous step
- `→`: Next step
- `Home`: First step
- `End`: Last step

**Configuration**:
- Adjust playback speed using the speed slider (0.5x - 4x)
- Set step size (1, 10, 100 data points)
- Enable loop mode to continuously cycle through data

**GIF Placeholder**:
```
[TODO: Add animated GIF demonstrating step controls in action]
```

## Testing

### Backend Tests

The backend uses pytest for testing. Tests are located in the `backend/tests/` directory.

**Run all tests**:
```bash
cd backend
poetry run pytest
```

**Run with coverage**:
```bash
poetry run pytest --cov=app --cov-report=html
```

**Run specific test file**:
```bash
poetry run pytest tests/test_api.py
```

**Run tests with verbose output**:
```bash
poetry run pytest -v
```

### Frontend Tests

The frontend uses standard npm testing tools.

**Linting**:
```bash
cd frontend
npm run lint
```

**Fix linting issues automatically**:
```bash
npm run lint:fix
```

**Run unit tests**:
```bash
npm run test
```

**Run tests in watch mode**:
```bash
npm run test:watch
```

**Run tests with coverage**:
```bash
npm run test:coverage
```

## Troubleshooting

### Common Issues

#### Backend Issues

**Issue**: `poetry: command not found`
```
Solution: Install Poetry using the official installer:
curl -sSL https://install.python-poetry.org | python3 -
Add Poetry to your PATH as instructed by the installer.
```

**Issue**: `ModuleNotFoundError` when running uvicorn
```
Solution: Ensure you're in the Poetry virtual environment:
poetry shell
If the issue persists, reinstall dependencies:
poetry install --no-cache
```

**Issue**: Port 8000 already in use
```
Solution: Either stop the process using port 8000:
lsof -ti:8000 | xargs kill -9
Or run the server on a different port:
poetry run uvicorn app.main:app --reload --port 8001
```

**Issue**: CORS errors when accessing from frontend
```
Solution: Verify ALLOWED_ORIGINS in backend/.env includes your frontend URL:
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
Restart the backend server after changes.
```

#### Frontend Issues

**Issue**: `npm install` fails with permission errors
```
Solution: Avoid using sudo. Instead, configure npm to use a different directory:
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
Add export PATH=~/.npm-global/bin:$PATH to ~/.profile
```

**Issue**: Frontend can't connect to API
```
Solution: Check VITE_API_BASE_URL in frontend/.env matches backend URL:
VITE_API_BASE_URL=http://localhost:8000
Verify the backend server is running and accessible.
```

**Issue**: Blank page or build errors
```
Solution: Clear cache and reinstall dependencies:
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**Issue**: Visualization not rendering
```
Solution: 
1. Check browser console for JavaScript errors
2. Verify data format matches expected schema
3. Try with a smaller dataset to rule out performance issues
4. Ensure WebGL is enabled in browser (for 3D visualizations)
```

#### Data/Performance Issues

**Issue**: Large dataset upload fails
```
Solution:
1. Check payload size against MAX_PAYLOAD_SIZE limit
2. Consider splitting data into multiple uploads
3. Use compression (gzip) for large payloads
4. Verify timeout settings in both frontend and backend
```

**Issue**: Slow visualization rendering
```
Solution:
1. Reduce VITE_MAX_DATA_POINTS in frontend/.env
2. Use data aggregation in macro view
3. Enable pagination for large datasets
4. Check browser performance/memory usage
```

### Getting Help

If you encounter issues not covered here:

1. Check the [Issues](../../issues) page for similar problems
2. Review API logs for error details (backend console output)
3. Check browser console for frontend errors
4. Enable DEBUG mode in backend/.env for detailed logging
5. Create a new issue with:
   - Detailed description of the problem
   - Steps to reproduce
   - Error messages and logs
   - Environment details (OS, Python version, Node version)

## Contributing

We welcome contributions! To get started:

1. **Fork the repository** and clone your fork
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following the existing code style
4. **Run tests** to ensure nothing breaks:
   ```bash
   # Backend
   cd backend && poetry run pytest
   
   # Frontend
   cd frontend && npm run lint && npm run test
   ```
5. **Commit your changes** with clear, descriptive messages
6. **Push to your fork** and create a pull request

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint configuration for JavaScript/TypeScript
- Write tests for new features
- Update documentation for API changes
- Keep commits atomic and well-described

### Code Review Process

All pull requests require:
- Passing CI/CD checks (tests, linting)
- Code review approval from maintainer
- Documentation updates (if applicable)
- No merge conflicts with main branch

---

**License**: [Add your license here]

**Maintainers**: [Add maintainer information]

**Version**: 1.0.0
