"""
Nemotron Client — Talks to a local vLLM server running NVIDIA Nemotron.

vLLM exposes an OpenAI-compatible API, so we use the ``openai`` Python
SDK pointed at the local endpoint.  This client powers the Local Guardian
agent's reasoning and tool-calling loop.
"""

from __future__ import annotations

from openai import OpenAI

from config.settings import NEMOTRON_ENDPOINT, NEMOTRON_MODEL


def get_client() -> OpenAI:
    """
    Return an OpenAI-compatible client pointed at the local vLLM server.

    The API key is ignored by vLLM but the SDK requires *something*.
    """
    return OpenAI(
        base_url=NEMOTRON_ENDPOINT,
        api_key="not-needed",
    )


def get_model() -> str:
    """Return the configured Nemotron model identifier."""
    return NEMOTRON_MODEL
