"""
FastAPI web application for the Deepfake Detection Agent.

Provides:
- GET  /              Upload page
- POST /scan          Start a new scan (accepts image upload)
- GET  /scan/{id}/stream  SSE endpoint for real-time progress
- GET  /report/{id}   View completed report
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from agent.orchestrator import DeepfakeDetectionAgent, ScanProgress
from core.config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Deepfake Detection Agent",
    description="Upload a photo to scan the web for AI deepfakes of you.",
    version="1.0.0",
)

# Static files and templates
_project_root = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=_project_root / "static"), name="static")
app.mount("/evidence", StaticFiles(directory=settings.evidence_dir), name="evidence")
templates = Jinja2Templates(directory=_project_root / "templates")

# In-memory scan state (for SSE streaming)
_active_scans: dict[str, asyncio.Queue] = {}
_scan_reports: dict[str, dict[str, Any]] = {}

# Agent instance
_agent = DeepfakeDetectionAgent()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/scan")
async def start_scan(file: UploadFile = File(...)):
    """
    Start a new deepfake detection scan.

    Accepts an image upload, saves it, and kicks off the agent scan
    in the background.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # Generate scan ID and save uploaded file
    scan_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    upload_path = settings.upload_dir / f"{scan_id}{ext}"

    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info("Scan %s: image saved to %s (%d bytes)", scan_id, upload_path, len(content))

    # Create progress queue for SSE streaming
    queue: asyncio.Queue[ScanProgress | None] = asyncio.Queue()
    _active_scans[scan_id] = queue

    # Start scan in background
    asyncio.create_task(_run_scan_background(scan_id, str(upload_path), queue))

    return JSONResponse({
        "scan_id": scan_id,
        "stream_url": f"/scan/{scan_id}/stream",
        "report_url": f"/report/{scan_id}",
    })


async def _run_scan_background(
    scan_id: str,
    image_path: str,
    queue: asyncio.Queue,
):
    """Run the agent scan and push progress events to the SSE queue."""
    try:
        async for progress in _agent.run_scan(image_path, scan_id):
            await queue.put(progress)

            # If scan is complete, check for report
            if progress.status == "complete":
                report_path = settings.output_dir / f"report_{scan_id}.json"
                if report_path.exists():
                    with open(report_path) as f:
                        _scan_reports[scan_id] = json.load(f)

    except Exception as e:
        logger.error("Background scan %s failed: %s", scan_id, e, exc_info=True)
        await queue.put(
            ScanProgress(
                scan_id=scan_id,
                status="error",
                phase="fatal",
                message=f"Scan failed unexpectedly: {str(e)}",
            )
        )
    finally:
        await queue.put(None)  # Signal end of stream


@app.get("/scan/{scan_id}/stream")
async def scan_stream(scan_id: str):
    """SSE endpoint for real-time scan progress."""
    if scan_id not in _active_scans:
        raise HTTPException(status_code=404, detail="Scan not found.")

    queue = _active_scans[scan_id]

    async def event_generator():
        while True:
            progress = await queue.get()
            if progress is None:
                # Send final event and close
                yield {
                    "event": "complete",
                    "data": json.dumps({"scan_id": scan_id, "status": "done"}),
                }
                break

            yield {
                "event": progress.status,
                "data": json.dumps(progress.to_event()),
            }

        # Cleanup
        _active_scans.pop(scan_id, None)

    return EventSourceResponse(event_generator())


@app.get("/report/{scan_id}", response_class=HTMLResponse)
async def view_report(request: Request, scan_id: str):
    """View the completed threat report."""
    # Try in-memory first, then disk
    report = _scan_reports.get(scan_id)

    if not report:
        report_path = settings.output_dir / f"report_{scan_id}.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            _scan_reports[scan_id] = report

    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "report": report,
            "scan_id": scan_id,
        },
    )


@app.get("/api/report/{scan_id}")
async def get_report_json(scan_id: str):
    """Get the report as raw JSON."""
    report = _scan_reports.get(scan_id)

    if not report:
        report_path = settings.output_dir / f"report_{scan_id}.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    return JSONResponse(report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )
