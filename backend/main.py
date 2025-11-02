"""
FastAPI application for LLM Visualization Advanced Backend
"""

from fastapi import FastAPI

app = FastAPI(
    title="LLM Visualization Advanced API",
    description="Backend API for LLM visualization and analysis",
    version="0.1.0"
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "LLM Visualization Advanced API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}