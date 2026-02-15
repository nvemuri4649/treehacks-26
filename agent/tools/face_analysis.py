"""
MCP Tool: Face Analysis using InsightFace.

Compares a candidate image's face against the reference embedding
to determine if it's the same person.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from claude_agent_sdk import tool

from core.face_engine import FaceEngine, FaceComparisonResult

logger = logging.getLogger(__name__)

# Singleton face engine instance
_face_engine = FaceEngine()

# In-memory storage for the reference embedding (set when scan starts)
_reference_embedding: np.ndarray | None = None


def set_reference_embedding(embedding: np.ndarray) -> None:
    """Set the reference face embedding for the current scan session."""
    global _reference_embedding
    _reference_embedding = embedding
    logger.info("Reference embedding set (shape=%s)", embedding.shape)


def get_reference_embedding() -> np.ndarray | None:
    """Get the current reference embedding."""
    return _reference_embedding


@tool(
    "analyze_face_match",
    "Compare a candidate image against the reference face to determine if it shows the same person. "
    "Returns a similarity score and match verdict. Use this to filter candidates before running "
    "expensive deepfake analysis -- only images that match the person's face should be analyzed.",
    {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Local file path to the candidate image.",
            },
            "threshold": {
                "type": "number",
                "description": "Optional cosine distance threshold override (default: 0.35, lower = stricter).",
            },
        },
        "required": ["image_path"],
    },
)
async def analyze_face_match(args: dict[str, Any]) -> dict[str, Any]:
    image_path = args["image_path"]
    threshold = args.get("threshold")

    logger.info("Tool: analyze_face_match(path=%s)", image_path)

    if _reference_embedding is None:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "error": "No reference embedding set. The scan must extract the user's face first.",
                    }),
                }
            ]
        }

    try:
        # Detect faces in the candidate image
        detections = _face_engine.detect_faces(image_path)

        if not detections:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "match": False,
                            "reason": "No face detected in the candidate image.",
                            "faces_found": 0,
                        }),
                    }
                ]
            }

        # Compare each detected face against the reference
        best_match: FaceComparisonResult | None = None
        best_face_idx = -1

        for idx, det in enumerate(detections):
            comparison = FaceComparisonResult.from_embeddings(
                _reference_embedding, det.embedding, threshold
            )
            if best_match is None or comparison.cosine_distance < best_match.cosine_distance:
                best_match = comparison
                best_face_idx = idx

        result = {
            "match": best_match.is_same_person,
            "cosine_similarity": round(best_match.cosine_similarity, 4),
            "cosine_distance": round(best_match.cosine_distance, 4),
            "confidence": round(best_match.confidence, 4),
            "faces_found": len(detections),
            "best_face_index": best_face_idx,
            "image_path": image_path,
        }

        if best_match.is_same_person:
            result["summary"] = (
                f"MATCH: Face matches the reference person with "
                f"{best_match.confidence:.0%} confidence "
                f"(cosine distance: {best_match.cosine_distance:.4f}). "
                f"This image should be analyzed for deepfake indicators."
            )
        else:
            result["summary"] = (
                f"NO MATCH: Best face has cosine distance {best_match.cosine_distance:.4f} "
                f"(threshold: {threshold or 0.35}). Not the same person."
            )

        return {
            "content": [
                {"type": "text", "text": json.dumps(result, indent=2)}
            ]
        }

    except Exception as e:
        logger.error("analyze_face_match failed: %s", e)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "error": str(e),
                        "image_path": image_path,
                    }),
                }
            ]
        }
