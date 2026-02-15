"""
Cena — FastAPI application entry point.

Serves:
  - WebSocket + REST API for the native Cena macOS agent chat
  - Deepfake Detection Agent web UI under /deepfake

Run with:
    python -m server.main
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config.settings import HOST, PORT
from server.routes import router

app = FastAPI(
    title="Cena",
    description="Personal data privacy suite for the age of AI",
    version="0.1.0",
)

# ── Core agent routes (WebSocket chat, sessions) ─────────────────────────
app.include_router(router)

# ── Deepfake Detection Agent (optional — requires playwright) ────────────
try:
    from deepfake.routes import router as deepfake_router  # noqa: E402

    _deepfake_dir = Path(__file__).resolve().parent.parent / "deepfake"
    app.mount(
        "/deepfake/static",
        StaticFiles(directory=_deepfake_dir / "static"),
        name="deepfake-static",
    )
    _evidence_dir = Path(__file__).resolve().parent.parent / "output" / "deepfake" / "evidence"
    _evidence_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/deepfake/evidence",
        StaticFiles(directory=_evidence_dir),
        name="deepfake-evidence",
    )
    app.include_router(deepfake_router, prefix="/deepfake", tags=["deepfake"])
except ImportError as _e:
    import logging as _log
    _log.warning("Deepfake module unavailable (missing dependency: %s) — skipping", _e)


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
            "deepfake_scanner": "GET /deepfake/",
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
