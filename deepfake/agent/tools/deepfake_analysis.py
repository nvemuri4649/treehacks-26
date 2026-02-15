"""
MCP Tool: Deepfake Analysis.

Runs the full multi-signal deepfake detection pipeline on an image:
Claude Vision + EXIF metadata + frequency domain analysis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from claude_agent_sdk import tool

from deepfake.core.deepfake_detector import DeepfakeDetector

logger = logging.getLogger(__name__)

_detector = DeepfakeDetector()


@tool(
    "detect_deepfake",
    "Run multi-signal deepfake analysis on a candidate image. "
    "Combines three detection signals: (1) Claude Vision artifact analysis, "
    "(2) EXIF/metadata forensic inspection, (3) frequency domain spectral analysis. "
    "Returns a deepfake probability score (0-1) and threat level. "
    "Only use this on images that have already passed face matching (analyze_face_match).",
    {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Local file path to the image to analyze.",
            },
            "source_url": {
                "type": "string",
                "description": "Original URL where the image was found (for the report).",
            },
            "skip_vision": {
                "type": "boolean",
                "description": "If true, skip Claude Vision analysis to save API cost. Default: false.",
            },
        },
        "required": ["image_path"],
    },
)
async def detect_deepfake(args: dict[str, Any]) -> dict[str, Any]:
    image_path = args["image_path"]
    source_url = args.get("source_url", "")
    skip_vision = args.get("skip_vision", False)

    logger.info("Tool: detect_deepfake(path=%s, skip_vision=%s)", image_path, skip_vision)

    try:
        result = await _detector.analyze(
            image_path, source_url=source_url, skip_vision=skip_vision
        )

        output: dict[str, Any] = {
            "deepfake_probability": round(result.deepfake_probability, 4),
            "threat_level": result.threat_level,
            "image_path": result.image_path,
            "source_url": result.source_url,
        }

        #Vision signal details
        if result.vision:
            output["vision_analysis"] = {
                "is_likely_deepfake": result.vision.is_likely_deepfake,
                "confidence": round(result.vision.confidence, 4),
                "artifacts_found": result.vision.artifacts_found,
                "reasoning": result.vision.reasoning,
            }

        #Metadata signal details
        if result.metadata:
            output["metadata_analysis"] = {
                "score": round(result.metadata.score, 4),
                "has_camera_exif": result.metadata.has_camera_exif,
                "has_ai_signatures": result.metadata.has_ai_signatures,
                "ai_tool_detected": result.metadata.ai_tool_detected,
                "exif_stripped": result.metadata.exif_stripped,
                "suspicious_fields": result.metadata.suspicious_fields,
            }

        #Frequency signal details
        if result.frequency:
            output["frequency_analysis"] = {
                "score": round(result.frequency.score, 4),
                "spectral_energy_ratio": round(result.frequency.spectral_energy_ratio, 4),
                "has_periodic_artifacts": result.frequency.has_periodic_artifacts,
                "has_spectral_gap": result.frequency.has_spectral_gap,
                "details": result.frequency.details,
            }

        #Generate human-readable summary
        if result.threat_level in ("high", "critical"):
            output["summary"] = (
                f"HIGH RISK: This image has a {result.deepfake_probability:.0%} probability "
                f"of being a deepfake (threat level: {result.threat_level}). "
                f"This should be flagged in the report."
            )
        elif result.threat_level == "medium":
            output["summary"] = (
                f"MODERATE RISK: This image has a {result.deepfake_probability:.0%} probability "
                f"of being a deepfake. Manual review recommended."
            )
        else:
            output["summary"] = (
                f"LOW RISK: This image has a {result.deepfake_probability:.0%} probability "
                f"of being a deepfake. Likely authentic."
            )

        return {
            "content": [
                {"type": "text", "text": json.dumps(output, indent=2)}
            ]
        }

    except Exception as e:
        logger.error("detect_deepfake failed: %s", e)
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
