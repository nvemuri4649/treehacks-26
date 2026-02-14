"""
Claude Client — Uses the Claude Agent SDK for agentic cloud task execution,
with a direct Anthropic API fallback for multimodal (image) requests.

Text-only requests go through ``claude_agent_sdk.query()`` so Claude can
use its full tool suite (Read, Write, Bash, etc.) to actually *do* things
for the user (e.g. draft a document, file a form, run code).

Multimodal requests use the Anthropic Messages API directly because the
Agent SDK's ``query()`` interface is text-only.
"""

from __future__ import annotations

import base64
from typing import Optional

import anthropic
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ResultMessage,
)

from config.settings import ANTHROPIC_API_KEY

# ── System prompt shared by both paths ────────────────────────────────────

_CLOUD_SYSTEM_PROMPT = (
    "You are a helpful assistant.  The user's personal information has been "
    "replaced with privacy tokens (e.g. [PERSON_1], [ADDRESS_1]) for "
    "security.  Carry out the user's request using these tokens as "
    "placeholders.  Do NOT ask the user to reveal the real values."
)


# =========================================================================
# Text path — Claude Agent SDK  (agentic, can use tools)
# =========================================================================

async def send_text(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    conversation_history: Optional[list[dict]] = None,
    max_tokens: int = 4096,
) -> str:
    """
    Send a sanitized text prompt to Claude via the Agent SDK.

    Claude runs as a full agent and can use built-in tools (Read, Write,
    Bash, etc.) to perform real tasks for the user.

    Args:
        prompt:               The sanitized user message.
        model:                Ignored by the Agent SDK (it uses its own model
                              selection), but kept for interface compatibility.
        conversation_history: Optional prior messages for context.
        max_tokens:           Not directly used by Agent SDK; kept for compat.

    Returns:
        The assistant's text response.
    """
    # Fold conversation history into the prompt so Claude has context
    full_prompt = prompt
    if conversation_history:
        parts = []
        for msg in conversation_history:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        context = "\n".join(parts)
        full_prompt = (
            f"Previous conversation (for context):\n{context}\n\n"
            f"Current request:\n{prompt}"
        )

    options = ClaudeAgentOptions(
        system_prompt=_CLOUD_SYSTEM_PROMPT,
        max_turns=5,
    )

    final_response = ""
    async for message in query(prompt=full_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    final_response = block.text
        elif isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError(f"Claude Agent SDK error: {message.result}")

    return final_response


# =========================================================================
# Multimodal path — Direct Anthropic API  (image + text)
# =========================================================================

def _get_api_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file (see .env.example)."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


async def send_multimodal(
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    model: str = "claude-sonnet-4-20250514",
    conversation_history: Optional[list[dict]] = None,
    max_tokens: int = 4096,
) -> str:
    """
    Send a text + image message to Claude via the Anthropic Messages API.

    Falls back to the direct API because the Agent SDK's ``query()``
    interface does not natively accept image payloads.

    Args:
        prompt:               Sanitized user text.
        image_bytes:          Transformed image bytes (privacy-safe).
        mime_type:            Image MIME type.
        model:                Anthropic model identifier.
        conversation_history: Optional prior messages.
        max_tokens:           Max response tokens.

    Returns:
        The assistant's text response.
    """
    client = _get_api_client()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_b64,
            },
        },
        {"type": "text", "text": prompt},
    ]

    messages: list[dict] = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_content})

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_CLOUD_SYSTEM_PROMPT,
        messages=messages,
    )

    return "".join(
        block.text for block in response.content if block.type == "text"
    )
