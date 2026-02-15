"""
Agent orchestrator using the Claude Agent SDK.

Wires together all 6 MCP tools, configures the agent, and provides
a high-level interface for running deepfake detection scans.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
)

from deepfake.agent.system_prompt import SYSTEM_PROMPT
from deepfake.agent.tools.reverse_image_search import reverse_image_search
from deepfake.agent.tools.serp_search import web_search
from deepfake.agent.tools.face_analysis import (
    analyze_face_match,
    set_reference_embedding,
)
from deepfake.agent.tools.image_download import download_image
from deepfake.agent.tools.deepfake_analysis import detect_deepfake
from deepfake.agent.tools.report_generator import generate_report

from deepfake.core.face_engine import FaceEngine
from deepfake.core.config import settings

logger = logging.getLogger(__name__)


#---------------------------------------------------------------------------
#Scan state
#---------------------------------------------------------------------------

@dataclass
class ScanProgress:
    """Real-time scan progress for SSE streaming."""

    scan_id: str
    status: str = "initializing"  # initializing, searching, analyzing, complete, error
    phase: str = ""
    message: str = ""
    timestamp: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
            "details": self.details,
        }


@dataclass
class ScanResult:
    """Final result of a deepfake detection scan."""

    scan_id: str
    success: bool
    report: dict[str, Any] | None = None
    report_path: str = ""
    error: str = ""


#---------------------------------------------------------------------------
#Agent orchestrator
#---------------------------------------------------------------------------

class DeepfakeDetectionAgent:
    """
    Orchestrates the deepfake detection workflow using Claude Agent SDK.

    Creates an in-process MCP server with all 6 tools, configures the agent
    with the specialized system prompt, and runs the autonomous scan loop.
    """

    def __init__(self):
        self._face_engine = FaceEngine()

    def _create_mcp_server(self):
        """Create the MCP server with all deepfake detection tools."""
        return create_sdk_mcp_server(
            name="deepfake_detector",
            version="1.0.0",
            tools=[
                reverse_image_search,
                web_search,
                analyze_face_match,
                download_image,
                detect_deepfake,
                generate_report,
            ],
        )

    def _create_options(self, mcp_server) -> ClaudeAgentOptions:
        """Configure the Claude Agent with tools and permissions."""
        return ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"df": mcp_server},
            allowed_tools=[
                "mcp__df__reverse_image_search",
                "mcp__df__web_search",
                "mcp__df__analyze_face_match",
                "mcp__df__download_image",
                "mcp__df__detect_deepfake",
                "mcp__df__generate_report",
            ],
        )

    async def run_scan(
        self,
        image_path: str | Path,
        scan_id: str | None = None,
    ) -> AsyncGenerator[ScanProgress, None]:
        """
        Run a complete deepfake detection scan.

        This is an async generator that yields ScanProgress events
        for real-time streaming to the frontend.

        Args:
            image_path: path to the uploaded user photo.
            scan_id: optional scan identifier (auto-generated if not provided).

        Yields:
            ScanProgress events as the scan proceeds.
        """
        scan_id = scan_id or str(uuid.uuid4())[:8]
        image_path = Path(image_path)

        #--- Phase 0: Extract reference face embedding ---
        yield ScanProgress(
            scan_id=scan_id,
            status="initializing",
            phase="face_extraction",
            message="Extracting face embedding from uploaded photo...",
        )

        try:
            embedding = self._face_engine.extract_embedding(str(image_path))
            set_reference_embedding(embedding)
        except ValueError as e:
            yield ScanProgress(
                scan_id=scan_id,
                status="error",
                phase="face_extraction",
                message=f"Could not detect a face in the uploaded image: {e}",
            )
            return

        yield ScanProgress(
            scan_id=scan_id,
            status="initializing",
            phase="face_extraction",
            message="Face embedding extracted successfully. Starting agent...",
        )

        #--- Phase 1-6: Agent-driven search and analysis ---
        mcp_server = self._create_mcp_server()
        options = self._create_options(mcp_server)

        initial_prompt = (
            f"A user has uploaded their photo for a deepfake detection scan.\n\n"
            f"**Scan ID**: {scan_id}\n"
            f"**Image path**: {image_path}\n\n"
            f"The user's face embedding has already been extracted and loaded as the reference. "
            f"Begin the systematic search following your protocol:\n"
            f"1. Start with reverse_image_search on the image\n"
            f"2. Analyze results and identify the person if possible\n"
            f"3. Fan out targeted searches\n"
            f"4. Download and analyze candidate images\n"
            f"5. Generate the final threat report\n\n"
            f"Begin now."
        )

        yield ScanProgress(
            scan_id=scan_id,
            status="searching",
            phase="agent_start",
            message="Agent started. Beginning web search...",
        )

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(initial_prompt)

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                #Agent is providing reasoning/updates
                                yield ScanProgress(
                                    scan_id=scan_id,
                                    status="searching",
                                    phase="agent_reasoning",
                                    message=block.text[:500],
                                )

                            elif isinstance(block, ToolUseBlock):
                                #Agent is calling a tool
                                phase = self._tool_to_phase(block.name)
                                yield ScanProgress(
                                    scan_id=scan_id,
                                    status="analyzing" if "deepfake" in block.name else "searching",
                                    phase=phase,
                                    message=f"Running {block.name}...",
                                    details={
                                        "tool": block.name,
                                        "input_preview": str(block.input)[:200],
                                    },
                                )

            #Check if a report was generated
            report_path = settings.output_dir / f"report_{scan_id}.json"
            report = None
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)

            yield ScanProgress(
                scan_id=scan_id,
                status="complete",
                phase="done",
                message="Scan complete. Report generated.",
                details={
                    "report_path": str(report_path) if report else "",
                    "has_report": report is not None,
                },
            )

        except Exception as e:
            logger.error("Agent scan failed: %s", e, exc_info=True)
            yield ScanProgress(
                scan_id=scan_id,
                status="error",
                phase="agent_error",
                message=f"Scan failed: {str(e)}",
            )

    async def run_scan_blocking(
        self,
        image_path: str | Path,
        scan_id: str | None = None,
    ) -> ScanResult:
        """
        Run a complete scan and return the final result (non-streaming).

        Useful for programmatic usage and testing.
        """
        scan_id = scan_id or str(uuid.uuid4())[:8]
        last_progress = None

        async for progress in self.run_scan(image_path, scan_id):
            last_progress = progress
            logger.info("[%s] %s: %s", progress.phase, progress.status, progress.message)

        if last_progress and last_progress.status == "complete":
            report_path = settings.output_dir / f"report_{scan_id}.json"
            report = None
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)

            return ScanResult(
                scan_id=scan_id,
                success=True,
                report=report,
                report_path=str(report_path),
            )
        else:
            return ScanResult(
                scan_id=scan_id,
                success=False,
                error=last_progress.message if last_progress else "Unknown error",
            )

    @staticmethod
    def _tool_to_phase(tool_name: str) -> str:
        """Map tool names to user-friendly phase labels."""
        mapping = {
            "mcp__df__reverse_image_search": "reverse_search",
            "mcp__df__web_search": "web_search",
            "mcp__df__download_image": "downloading",
            "mcp__df__analyze_face_match": "face_matching",
            "mcp__df__detect_deepfake": "deepfake_analysis",
            "mcp__df__generate_report": "report_generation",
        }
        return mapping.get(tool_name, tool_name)
