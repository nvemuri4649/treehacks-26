"""
Cena — FastAPI application entry point.

Serves the WebSocket + REST API that the native Cena macOS agent
chat connects to. No web frontend — the UI is native Swift.

Run with:
    python -m server.main
"""

import uvicorn
from fastapi import FastAPI

from config.settings import HOST, PORT
from server.routes import router

app = FastAPI(
    title="Cena",
    description="Personal data privacy suite for the age of AI",
    version="0.1.0",
)

# Register API / WebSocket routes
app.include_router(router)


@app.get("/")
async def root():
    """API root — confirms the server is running."""
    return {
        "service": "cena",
        "status": "ok",
        "endpoints": {
            "health": "GET /health",
            "session": "POST /api/session",
            "chat": "WS /ws/{session_id}",
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )
