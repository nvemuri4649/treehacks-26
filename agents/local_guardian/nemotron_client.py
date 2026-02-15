"""
Nemotron Client — Talks to a local vLLM server running NVIDIA Nemotron.

vLLM exposes an OpenAI-compatible API, so we use the ``openai`` Python
SDK pointed at the local endpoint.  This client powers the Local Guardian
agent's reasoning and tool-calling loop.
"""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI

from config.settings import NEMOTRON_ENDPOINT, NEMOTRON_MODEL

#Generous timeout for local vLLM — first request may be slow (JIT compile)
_NEMOTRON_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)


def get_client() -> AsyncOpenAI:
    """
    Return an async OpenAI-compatible client pointed at the local vLLM server.

    The API key is ignored by vLLM but the SDK requires *something*.
    """
    return AsyncOpenAI(
        base_url=NEMOTRON_ENDPOINT,
        api_key="not-needed",
        timeout=_NEMOTRON_TIMEOUT,
    )


def get_model() -> str:
    """Return the configured Nemotron model identifier."""
    return NEMOTRON_MODEL
