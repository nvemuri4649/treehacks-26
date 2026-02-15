"""
MCP Tool: Reverse Image Search via Bright Data Scraping Browser.

Performs a Google Lens reverse image search to find visually similar
images across the web.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from claude_agent_sdk import tool

from core.search_engine import ReverseImageSearchEngine

logger = logging.getLogger(__name__)

_engine = ReverseImageSearchEngine()


@tool(
    "reverse_image_search",
    "Perform a reverse image search using Google Lens via Bright Data Scraping Browser. "
    "Finds web pages and images that are visually similar to the uploaded photo. "
    "Use this as the first search strategy to discover where the person's face appears online.",
    {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Local file path to the image to reverse-search.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 30).",
            },
        },
        "required": ["image_path"],
    },
)
async def reverse_image_search(args: dict[str, Any]) -> dict[str, Any]:
    image_path = args["image_path"]
    max_results = args.get("max_results", 30)

    logger.info("Tool: reverse_image_search(path=%s, max=%d)", image_path, max_results)

    try:
        results = await _engine.reverse_image_search(image_path, max_results=max_results)

        output = []
        for r in results:
            output.append({
                "page_url": r.page_url,
                "image_url": r.image_url,
                "title": r.title,
                "similarity": r.similarity_label,
            })

        summary = (
            f"Found {len(output)} visually similar results. "
            f"Examine the page URLs and titles to identify the person and "
            f"find any potentially manipulated images."
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "summary": summary,
                        "result_count": len(output),
                        "results": output,
                    }, indent=2),
                }
            ]
        }

    except Exception as e:
        logger.error("reverse_image_search failed: %s", e)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "error": str(e),
                        "summary": "Reverse image search failed. Try web_search as fallback.",
                    }),
                }
            ]
        }
