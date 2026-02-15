"""
API routes for Cena.

Endpoints:
  GET  /health              — Healthcheck
  POST /api/session         — Create a new chat session
  DELETE /api/session/{id}  — Delete a session
  WS   /ws/{session_id}     — WebSocket chat endpoint
"""

from __future__ import annotations

import json
import base64
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from agents.local_guardian.agent import create_session, delete_session, process_message
from config.settings import DEFAULT_CLOUD_MODEL

router = APIRouter()


# ------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------

@router.get("/health")
async def health() -> dict:
    """Simple healthcheck."""
    return {"status": "ok", "service": "cena"}


@router.post("/api/session")
async def new_session() -> dict:
    """Create a new chat session and return its ID."""
    session_id = create_session()
    return {"session_id": session_id}


@router.delete("/api/session/{session_id}")
async def remove_session(session_id: str) -> dict:
    """Delete a chat session and clean up its mappings."""
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


# ------------------------------------------------------------------
# WebSocket chat endpoint
# ------------------------------------------------------------------

@router.websocket("/ws/{session_id}")
async def websocket_chat(ws: WebSocket, session_id: str):
    """
    Real-time chat over WebSocket.

    Client sends JSON messages:
    {
        "text": "user message here",
        "model": "claude-sonnet-4-20250514",         // optional
        "image": "base64-encoded-image-data",  // optional
        "mime_type": "image/png"               // required if image present
    }

    Server responds with JSON:
    {
        "type": "status",
        "stage": "sanitizing" | "thinking" | "restoring"
    }
    ...then:
    {
        "type": "response",
        "text": "final response with PII restored",
        "sanitized_prompt": "what was sent to cloud",
        "model": "model-used",
        "had_image": false
    }
    ...or on error:
    {
        "type": "error",
        "message": "error description"
    }
    """
    await ws.accept()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            text = data.get("text", "").strip()
            if not text:
                await ws.send_json({"type": "error", "message": "No text provided"})
                continue

            model = data.get("model") or DEFAULT_CLOUD_MODEL
            image_b64 = data.get("image")
            mime_type = data.get("mime_type")

            # Decode base64 image if provided
            image_bytes: bytes | None = None
            if image_b64:
                try:
                    image_bytes = base64.b64decode(image_b64)
                except Exception:
                    await ws.send_json({
                        "type": "error",
                        "message": "Invalid base64 image data",
                    })
                    continue

            try:
                # Stage 1: Sanitizing (dereferencing PII locally)
                await ws.send_json({"type": "status", "stage": "sanitizing"})

                # Stage 1.5: Glazing (if image uploaded, apply adversarial protection)
                if image_bytes:
                    await ws.send_json({"type": "status", "stage": "glazing"})

                # Stage 2 & 3: Thinking (cloud call) + Restoring
                # The process_message function handles the full pipeline
                await ws.send_json({"type": "status", "stage": "thinking"})

                result = await process_message(
                    session_id=session_id,
                    text=text,
                    model=model,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                )

                await ws.send_json({"type": "status", "stage": "restoring"})

                # Send final response (include privacy report if available)
                response_payload = {
                    "type": "response",
                    "text": result["response"],
                    "sanitized_prompt": result["sanitized_prompt"],
                    "model": result["model"],
                    "had_image": result["had_image"],
                }
                if "privacy_report" in result:
                    response_payload["privacy_report"] = result["privacy_report"]

                await ws.send_json(response_payload)

            except Exception as e:
                traceback.print_exc()
                await ws.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}",
                })

    except WebSocketDisconnect:
        pass  # Client disconnected — nothing to clean up
