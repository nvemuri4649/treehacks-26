"""
MCP Tool: Report Generator.

Compiles findings from face matching and deepfake analysis into
a structured threat report.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

from core.config import settings

logger = logging.getLogger(__name__)


@tool(
    "generate_report",
    "Compile all findings into a structured deepfake threat report. "
    "Call this after completing all search, face matching, and deepfake analysis. "
    "Provide the list of flagged findings with their analysis results.",
    {
        "type": "object",
        "properties": {
            "scan_id": {
                "type": "string",
                "description": "The unique scan identifier.",
            },
            "person_name": {
                "type": "string",
                "description": "Identified name of the person (if discovered during search). Empty string if unknown.",
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_url": {"type": "string"},
                        "image_path": {"type": "string"},
                        "face_similarity": {"type": "number"},
                        "deepfake_probability": {"type": "number"},
                        "threat_level": {"type": "string"},
                        "artifacts": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "analysis_summary": {"type": "string"},
                    },
                },
                "description": "List of flagged images with their analysis results.",
            },
            "search_summary": {
                "type": "object",
                "properties": {
                    "reverse_image_results": {"type": "integer"},
                    "text_search_results": {"type": "integer"},
                    "image_search_results": {"type": "integer"},
                    "platform_results": {"type": "integer"},
                    "images_downloaded": {"type": "integer"},
                    "faces_matched": {"type": "integer"},
                    "images_analyzed": {"type": "integer"},
                },
                "description": "Summary statistics of the search effort.",
            },
        },
        "required": ["scan_id", "findings", "search_summary"],
    },
)
async def generate_report(args: dict[str, Any]) -> dict[str, Any]:
    scan_id = args["scan_id"]
    person_name = args.get("person_name", "Unknown")
    findings = args["findings"]
    search_summary = args["search_summary"]

    logger.info("Tool: generate_report(scan_id=%s, findings=%d)", scan_id, len(findings))

    # Determine overall threat level
    if not findings:
        overall_threat = "none"
        overall_message = (
            "No potential deepfakes were detected for this person. "
            "The search covered multiple platforms and image sources."
        )
    else:
        max_prob = max(f.get("deepfake_probability", 0) for f in findings)
        if max_prob >= 0.8:
            overall_threat = "critical"
            overall_message = (
                f"CRITICAL: {len(findings)} potential deepfake(s) detected with "
                f"high confidence. Immediate review recommended."
            )
        elif max_prob >= 0.6:
            overall_threat = "high"
            overall_message = (
                f"HIGH: {len(findings)} suspicious image(s) detected. "
                f"Several show significant deepfake indicators."
            )
        elif max_prob >= 0.4:
            overall_threat = "medium"
            overall_message = (
                f"MEDIUM: {len(findings)} image(s) flagged for review. "
                f"Some show moderate deepfake indicators."
            )
        else:
            overall_threat = "low"
            overall_message = (
                f"LOW: {len(findings)} image(s) showed minor indicators "
                f"but are likely authentic."
            )

    # Sort findings by deepfake probability descending
    sorted_findings = sorted(
        findings,
        key=lambda f: f.get("deepfake_probability", 0),
        reverse=True,
    )

    # Build the report
    report = {
        "scan_id": scan_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "person_name": person_name,
        "overall_threat_level": overall_threat,
        "overall_message": overall_message,
        "statistics": {
            **search_summary,
            "total_findings": len(findings),
            "critical_findings": sum(
                1 for f in findings if f.get("threat_level") == "critical"
            ),
            "high_findings": sum(
                1 for f in findings if f.get("threat_level") == "high"
            ),
        },
        "findings": sorted_findings,
    }

    # Save report to disk
    report_path = settings.output_dir / f"report_{scan_id}.json"
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved: %s", report_path)
    except Exception as e:
        logger.error("Failed to save report: %s", e)

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps({
                    "summary": overall_message,
                    "report_path": str(report_path),
                    "report": report,
                }, indent=2),
            }
        ]
    }
