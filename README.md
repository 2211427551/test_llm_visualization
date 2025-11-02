# LLM Visualization Advanced

A comprehensive web application for visualizing and analyzing Large Language Models with advanced interactive components.

## Project Structure

```
llm-viz-advanced/
├── backend/                 # FastAPI Python backend
│   ├── app/                # Application package
│   │   ├── __init__.py
│   │   └── main.py         # FastAPI application entry point
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Docker configuration
├── frontend/               # Svelte + TypeScript frontend
│   ├── src/
│   │   ├── routes/        # SvelteKit routes
│   │   ├── app.html       # HTML template
│   │   └── app.css        # Global styles
│   ├── static/            # Static assets
│   ├── package.json       # Node.js dependencies
│   ├── svelte.config.js   # SvelteKit configuration
│   ├── tsconfig.json      # TypeScript configuration
│   ├── vite.config.ts     # Vite configuration
│   ├── tailwind.config.js # Tailwind CSS configuration
│   └── postcss.config.js  # PostCSS configuration
└── README.md              # This file
```

## Development Setup

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

The backend API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

The frontend application will be available at `http://localhost:5173`

### Docker Setup

#### Backend Docker

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t llm-viz-backend .
   docker run -p 8000:8000 llm-viz-backend
   ```

## Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **PyTorch**: Machine learning framework (placeholder)
- **Transformers**: NLP library for transformer models (placeholder)
- **NumPy & Pandas**: Data processing libraries
- **Matplotlib & Plotly**: Visualization libraries

### Frontend
- **SvelteKit**: Full-stack web framework built on Svelte
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Vite**: Build tool and development server
- **D3.js**: Data visualization library

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check endpoint

## Contributing

This project is currently under development. Please refer to the project documentation for contribution guidelines.

## License

[Add license information here]