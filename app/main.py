# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes_scan import router as scan_router

app = FastAPI(
    title="TrueSight Backend v2",
    version="0.1.0",
    description="FastAPI backend for TrueSight v2 (PyTorch-based detector).",
)

# CORS – allow extension + local dev domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "truesight-backend-v2", "version": "0.1.0"}


# Mount v1 API routes
app.include_router(scan_router, prefix="/api/v1")
