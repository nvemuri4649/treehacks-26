"""
OpenAI Client — Async wrapper around the OpenAI Chat Completions API.

Supports both text-only and multimodal (text + image) requests via GPT models.
"""

from __future__ import annotations

import base64
from typing import Optional

import openai

from config.settings import OPENAI_API_KEY


def _get_client() -> openai.OpenAI:
    """Return a configured OpenAI client."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file (see .env.example)."
        )
    return openai.OpenAI(api_key=OPENAI_API_KEY)


async def send_text(
    prompt: str,
    model: str = "gpt-4o",
    conversation_history: Optional[list[dict]] = None,
    max_tokens: int = 4096,
) -> str:
    """
    Send a text-only message to an OpenAI GPT model and return the response.

    Args:
        prompt:               The sanitized user message.
        model:                OpenAI model identifier.
        conversation_history: Optional list of prior messages for multi-turn.
                              Each item: {"role": "user"|"assistant", "content": str}
        max_tokens:           Maximum tokens in the response.

    Returns:
        The assistant's text response.
    """
    client = _get_client()

    messages: list[dict] = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )

    return response.choices[0].message.content or ""


async def send_multimodal(
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    model: str = "gpt-4o",
    conversation_history: Optional[list[dict]] = None,
    max_tokens: int = 4096,
) -> str:
    """
    Send a text + image message to an OpenAI GPT model and return the response.

    Args:
        prompt:               The sanitized user text.
        image_bytes:          Transformed image bytes (already privacy-safe).
        mime_type:            Image MIME type (e.g. "image/png", "image/jpeg").
        model:                OpenAI model identifier (must support vision).
        conversation_history: Optional prior messages for multi-turn context.
        max_tokens:           Maximum tokens in the response.

    Returns:
        The assistant's text response.
    """
    client = _get_client()

    # Encode image as base64 data URL
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_b64}"

    # Build the multimodal user message
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
        {
            "type": "text",
            "text": prompt,
        },
    ]

    messages: list[dict] = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )

    return response.choices[0].message.content or ""
