"""
MCP Tool: Web Search via Bright Data SERP API.

Performs Google text and image searches to find deepfake-related
content mentioning the identified person.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from claude_agent_sdk import tool

from deepfake.core.search_engine import SerpSearchEngine, PlatformCrawler

logger = logging.getLogger(__name__)

_serp = SerpSearchEngine()
_platforms = PlatformCrawler()


@tool(
    "web_search",
    "Search the web using Google via Bright Data SERP API. "
    "Supports text search and image search. Use this to search for deepfake-related "
    "content by name (e.g., '[name] deepfake', '[name] AI generated face'). "
    "Also supports platform-specific searches on Reddit and Twitter/X.",
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "search_type": {
                "type": "string",
                "description": "Type of search: 'text', 'images', or 'platform'. Default: 'text'.",
                "enum": ["text", "images", "platform"],
            },
            "platform": {
                "type": "string",
                "description": "Platform name for platform search (reddit, twitter). Required when search_type='platform'.",
                "enum": ["reddit", "twitter"],
            },
            "num_results": {
                "type": "integer",
                "description": "Maximum number of results (default 20).",
            },
        },
        "required": ["query"],
    },
)
async def web_search(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"]
    search_type = args.get("search_type", "text")
    num_results = args.get("num_results", 20)

    logger.info("Tool: web_search(q='%s', type=%s, n=%d)", query, search_type, num_results)

    try:
        if search_type == "images":
            results = await _serp.search_images(query, num_results=num_results)
            output = [
                {
                    "image_url": r.image_url,
                    "source_url": r.source_url,
                    "title": r.title,
                    "width": r.width,
                    "height": r.height,
                }
                for r in results
            ]
            summary = f"Image search for '{query}' returned {len(output)} results."

        elif search_type == "platform":
            platform = args.get("platform", "reddit")
            results = await _platforms.crawl_platform(platform, query, max_results=num_results)
            output = [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                }
                for r in results
            ]
            summary = f"Platform search on {platform} for '{query}' returned {len(output)} results."

        else:  # text search
            results = await _serp.search_text(query, num_results=num_results)
            output = [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "position": r.position,
                }
                for r in results
            ]
            summary = f"Text search for '{query}' returned {len(output)} results."

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "summary": summary,
                        "query": query,
                        "search_type": search_type,
                        "result_count": len(output),
                        "results": output,
                    }, indent=2),
                }
            ]
        }

    except Exception as e:
        logger.error("web_search failed: %s", e)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "error": str(e),
                        "query": query,
                        "search_type": search_type,
                    }),
                }
            ]
        }
