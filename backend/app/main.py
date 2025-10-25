from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Data Visualization API",
    description="Backend API for model execution visualization",
    version="0.1.0"
)

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Data Visualization API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "model_forward": "/model/forward",
            "docs": "/docs"
        }
    }
