"""
Cena — FastAPI application entry point.

Run with:
    python -m server.main
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config.settings import HOST, PORT
from server.routes import router

app = FastAPI(
    title="Cena",
    description="Personal data privacy suite for the age of AI",
    version="0.1.0",
)

# Register API / WebSocket routes
app.include_router(router)

# Serve frontend static files
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.get("/")
async def serve_index():
    """Serve the chatbot UI."""
    return FileResponse(str(_frontend_dir / "index.html"))


if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )
