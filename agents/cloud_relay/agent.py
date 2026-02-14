"""
Cloud Relay Agent — Routes sanitized requests to the appropriate cloud LLM.

This is Agent 2.  In the Claude Agent SDK architecture it is surfaced as:
  • An MCP tool server ("cloud") with the ``send_to_cloud_llm`` tool.
  • An ``AgentDefinition`` subagent named ``cloud_relay`` so the Local
    Guardian can delegate to it.

Both representations use this module's ``relay()`` helper, which routes to
the correct provider client (Anthropic Claude or OpenAI GPT).

Supported providers:
  - Anthropic Claude  (claude-sonnet-4-20250514, claude-haiku-4-20250414, …)
  - OpenAI GPT        (gpt-4o, gpt-4o-mini, …)
"""

from __future__ import annotations

from typing import Optional

from config.settings import DEFAULT_CLOUD_MODEL, get_provider
from agents.cloud_relay import claude_client, openai_client


async def relay(
    prompt: str,
    model: str | None = None,
    image_bytes: bytes | None = None,
    mime_type: str | None = None,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Send a sanitized request to a cloud LLM and return the raw response.

    This is the core routing function used by the ``send_to_cloud_llm`` MCP
    tool defined in ``agents.local_guardian.agent``.

    Args:
        prompt:               Sanitized user text (PII already redacted).
        model:                Model identifier (e.g. "claude-sonnet-4-20250514",
                              "gpt-4o").  Falls back to DEFAULT_MODEL.
        image_bytes:          Optional transformed image bytes.
        mime_type:            Required if image_bytes is provided.
        conversation_history: Optional multi-turn history list.

    Returns:
        The LLM's text response (still containing redaction tokens).

    Raises:
        ValueError:    If the model name is unrecognised.
        RuntimeError:  If the required API key is missing.
    """
    model = model or DEFAULT_CLOUD_MODEL
    provider = get_provider(model)
    has_image = image_bytes is not None and len(image_bytes) > 0

    if provider == "claude":
        if has_image:
            return await claude_client.send_multimodal(
                prompt=prompt,
                image_bytes=image_bytes,
                mime_type=mime_type or "image/png",
                model=model,
                conversation_history=conversation_history,
            )
        else:
            return await claude_client.send_text(
                prompt=prompt,
                model=model,
                conversation_history=conversation_history,
            )

    elif provider == "openai":
        if has_image:
            return await openai_client.send_multimodal(
                prompt=prompt,
                image_bytes=image_bytes,
                mime_type=mime_type or "image/png",
                model=model,
                conversation_history=conversation_history,
            )
        else:
            return await openai_client.send_text(
                prompt=prompt,
                model=model,
                conversation_history=conversation_history,
            )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
