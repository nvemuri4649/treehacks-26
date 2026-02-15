"""
MCP Tool: Image Download.

Downloads candidate images from URLs, validates them, and saves
locally for face matching and deepfake analysis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from claude_agent_sdk import tool

from deepfake.core.image_processor import ImageProcessor

logger = logging.getLogger(__name__)

_processor = ImageProcessor()


@tool(
    "download_image",
    "Download an image from a URL for analysis. The image is validated, "
    "deduplicated by content hash, resized, and saved locally. "
    "Returns the local file path for use with analyze_face_match and detect_deepfake. "
    "Supports batch downloading of multiple URLs.",
    {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of image URLs to download. Can be a single URL in a list.",
            },
        },
        "required": ["urls"],
    },
)
async def download_image(args: dict[str, Any]) -> dict[str, Any]:
    urls = args["urls"]

    logger.info("Tool: download_image(count=%d)", len(urls))

    try:
        downloaded, stats = await _processor.download_batch(urls)

        results = []
        for dl in downloaded:
            results.append({
                "source_url": dl.source_url,
                "local_path": str(dl.local_path),
                "content_hash": dl.content_hash,
                "width": dl.width,
                "height": dl.height,
                "format": dl.format,
                "file_size_kb": round(dl.file_size / 1024, 1),
            })

        summary = (
            f"Downloaded {stats.successful}/{stats.total_attempted} images. "
            f"Duplicates skipped: {stats.duplicates_skipped}. "
            f"Failed: {stats.failed}."
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "summary": summary,
                        "stats": {
                            "total_attempted": stats.total_attempted,
                            "successful": stats.successful,
                            "duplicates_skipped": stats.duplicates_skipped,
                            "failed": stats.failed,
                        },
                        "downloaded": results,
                    }, indent=2),
                }
            ]
        }

    except Exception as e:
        logger.error("download_image failed: %s", e)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"error": str(e)}),
                }
            ]
        }
