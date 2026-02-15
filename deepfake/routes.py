"""
Deepfake Detection routes — mountable FastAPI router.

Endpoints (all relative to the mount prefix, e.g. /deepfake):
  GET  /              Upload page
  POST /scan          Start a new scan (accepts image upload)
  GET  /scan/{id}/stream  SSE endpoint for real-time progress
  GET  /report/{id}   View completed report
  GET  /api/report/{id}  JSON report data
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from deepfake.agent.orchestrator import DeepfakeDetectionAgent, ScanProgress
from deepfake.core.config import settings

logger = logging.getLogger(__name__)

#---------------------------------------------------------------------------
#Router + templates
#---------------------------------------------------------------------------

router = APIRouter()

_templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=_templates_dir)

#In-memory scan state
_active_scans: dict[str, asyncio.Queue] = {}
_scan_reports: dict[str, dict[str, Any]] = {}

#Lazy-initialised agent (avoid heavy imports at module load)
_agent: DeepfakeDetectionAgent | None = None


def _get_agent() -> DeepfakeDetectionAgent:
    global _agent
    if _agent is None:
        _agent = DeepfakeDetectionAgent()
    return _agent


#---------------------------------------------------------------------------
#Routes
#---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/scan")
async def start_scan(file: UploadFile = File(...)):
    """Start a new deepfake detection scan."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    scan_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    upload_path = settings.upload_dir / f"{scan_id}{ext}"

    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    logger.info("Scan %s: image saved to %s (%d bytes)", scan_id, upload_path, len(content))

    queue: asyncio.Queue[ScanProgress | None] = asyncio.Queue()
    _active_scans[scan_id] = queue

    asyncio.create_task(_run_scan_background(scan_id, str(upload_path), queue))

    return JSONResponse({
        "scan_id": scan_id,
        "stream_url": f"/deepfake/scan/{scan_id}/stream",
        "report_url": f"/deepfake/report/{scan_id}",
    })


async def _run_scan_background(
    scan_id: str,
    image_path: str,
    queue: asyncio.Queue,
):
    """Run the agent scan and push progress events to the SSE queue."""
    try:
        agent = _get_agent()
        async for progress in agent.run_scan(image_path, scan_id):
            await queue.put(progress)

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
        await queue.put(None)


@router.get("/scan/{scan_id}/stream")
async def scan_stream(scan_id: str):
    """SSE endpoint for real-time scan progress."""
    if scan_id not in _active_scans:
        raise HTTPException(status_code=404, detail="Scan not found.")

    queue = _active_scans[scan_id]

    async def event_generator():
        while True:
            progress = await queue.get()
            if progress is None:
                yield {
                    "event": "complete",
                    "data": json.dumps({"scan_id": scan_id, "status": "done"}),
                }
                break

            yield {
                "event": progress.status,
                "data": json.dumps(progress.to_event()),
            }

        _active_scans.pop(scan_id, None)

    return EventSourceResponse(event_generator())


@router.get("/report/{scan_id}", response_class=HTMLResponse)
async def view_report(request: Request, scan_id: str):
    """View the completed threat report."""
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
        {"request": request, "report": report, "scan_id": scan_id},
    )


@router.get("/api/report/{scan_id}")
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
