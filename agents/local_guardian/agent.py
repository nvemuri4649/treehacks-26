"""
Local Guardian Agent — Privacy-first orchestrator powered by NVIDIA Nemotron.

Nemotron runs locally via vLLM (OpenAI-compatible API) and drives the full
privacy pipeline through function-calling / tool use:

  1. ``redact_text``      — strip PII from the user's message
  2. ``transform_image``  — anonymise an uploaded image
  3. ``send_to_cloud``    — forward sanitized data to the Cloud Relay
  4. ``re_reference_text`` — restore PII tokens in the cloud response

After the Cloud Relay (Claude Agent SDK or OpenAI GPT) returns, Nemotron
re-references the tokens and produces the final user-facing answer.

All personal data stays on the local machine.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from agents.local_guardian.nemotron_client import get_client as get_nemotron, get_model as get_nemotron_model
from agents.local_guardian.redactor import redact
from agents.local_guardian.image_transformer import transform
from agents.local_guardian.rereferencer import re_reference
from agents.local_guardian.mapping_store import store as mapping_store
from agents.cloud_relay.agent import relay as cloud_relay
from config.settings import DEFAULT_CLOUD_MODEL


# =========================================================================
# Tool definitions (OpenAI function-calling schema)
# =========================================================================

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "redact_text",
            "description": (
                "Redact personal information from user text, replacing PII "
                "with privacy tokens like [PERSON_1], [ADDRESS_1], etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The raw user text that may contain PII.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Current chat session identifier.",
                    },
                },
                "required": ["text", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transform_image",
            "description": (
                "Transform an uploaded image to remove or obfuscate personal "
                "information before sending it to the cloud."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Current chat session identifier.",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_cloud",
            "description": (
                "Send a sanitized (redacted) prompt to a cloud LLM "
                "(Claude via Claude Agent SDK, or OpenAI GPT) and return "
                "the response.  ONLY pass already-redacted text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sanitized_prompt": {
                        "type": "string",
                        "description": "The redacted user text with PII tokens.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Cloud model identifier, e.g. 'claude-sonnet-4-20250514' or 'gpt-4o'.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Current chat session identifier.",
                    },
                    "include_image": {
                        "type": "boolean",
                        "description": "Whether to include the transformed image.",
                    },
                },
                "required": ["sanitized_prompt", "model", "session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "re_reference_text",
            "description": (
                "Restore original personal information in text by replacing "
                "privacy tokens (e.g. [PERSON_1]) back to real values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text containing privacy tokens to restore.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Current chat session identifier.",
                    },
                },
                "required": ["text", "session_id"],
            },
        },
    },
]


# =========================================================================
# System prompt for the local Nemotron agent
# =========================================================================

GUARDIAN_SYSTEM_PROMPT = """\
You are the Cena Local Guardian agent running on the user's own \
machine.  Your sole purpose is to protect user privacy by processing every \
message through a strict pipeline before and after cloud LLM calls.

For EVERY user message, follow these steps IN ORDER:

1. REDACT — Call `redact_text` with the raw user message and session_id to \
   replace personal information with tokens.
2. TRANSFORM IMAGE — If the metadata says "Has Image: True", call \
   `transform_image` with the session_id.
3. CLOUD RELAY — Call `send_to_cloud` with the SANITIZED text from step 1, \
   the requested cloud model, the session_id, and include_image=true if \
   there was an image.
4. RE-REFERENCE — Call `re_reference_text` with the cloud response text \
   and the session_id to restore original personal information.
5. RETURN — Output ONLY the re-referenced text from step 4 as your final \
   answer.  Do NOT add any of your own commentary.

RULES:
- NEVER skip any step.
- NEVER send un-redacted text to send_to_cloud.
- NEVER reveal the pipeline mechanics to the user.
- Your final text output must be ONLY the restored cloud response.\
"""


# =========================================================================
# Session state
# =========================================================================

@dataclass
class Session:
    """Holds state for a single chat session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cloud_history: list[dict] = field(default_factory=list)


_sessions: dict[str, Session] = {}
_pending_images: dict[str, dict[str, Any]] = {}
_last_sanitized: dict[str, str] = {}


# =========================================================================
# Tool execution  (called inside the Nemotron tool-calling loop)
# =========================================================================

async def _execute_tool(
    name: str,
    args: dict[str, Any],
) -> str:
    """
    Execute a tool locally and return the result as a JSON string.
    """
    if name == "redact_text":
        sanitized, new_mapping = redact(args["text"], args["session_id"])
        _last_sanitized[args["session_id"]] = sanitized
        return json.dumps({
            "sanitized_text": sanitized,
            "new_tokens_count": len(new_mapping),
        })

    if name == "transform_image":
        session_id = args["session_id"]
        img = _pending_images.get(session_id)
        if not img or not img.get("raw"):
            return json.dumps({"status": "no_image", "message": "No image pending."})
        transformed = transform(img["raw"], img["mime_type"])
        img["transformed"] = transformed
        return json.dumps({"status": "ok", "bytes": len(transformed)})

    if name == "send_to_cloud":
        session_id = args["session_id"]
        model = args.get("model") or DEFAULT_CLOUD_MODEL
        include_image = args.get("include_image", False)

        image_bytes: bytes | None = None
        mime_type: str | None = None
        if include_image:
            img = _pending_images.get(session_id, {})
            image_bytes = img.get("transformed") or img.get("raw")
            mime_type = img.get("mime_type")

        session = _sessions.get(session_id)
        history = session.cloud_history if session else []

        response = await cloud_relay(
            prompt=args["sanitized_prompt"],
            model=model,
            image_bytes=image_bytes,
            mime_type=mime_type,
            conversation_history=history or None,
        )

        if session:
            session.cloud_history.append({"role": "user", "content": args["sanitized_prompt"]})
            session.cloud_history.append({"role": "assistant", "content": response})

        return response  # plain text — the cloud LLM's answer

    if name == "re_reference_text":
        mapping = mapping_store.get_mapping(args["session_id"])
        restored = re_reference(args["text"], mapping)
        return restored  # plain text — the final user-facing answer

    return json.dumps({"error": f"Unknown tool: {name}"})


# =========================================================================
# Nemotron tool-calling loop
# =========================================================================

async def _run_nemotron_pipeline(
    agent_prompt: str,
    max_iterations: int = 10,
) -> str:
    """
    Send *agent_prompt* to the local Nemotron model and execute any tool
    calls it requests until it produces a final text response.

    Returns:
        The model's final text answer (should be the re-referenced response).
    """
    client = get_nemotron()
    model = get_nemotron_model()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": GUARDIAN_SYSTEM_PROMPT},
        {"role": "user", "content": agent_prompt},
    ]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        choice = response.choices[0]

        # If the model is done (no more tool calls), return its text.
        if choice.finish_reason != "tool_calls" and not choice.message.tool_calls:
            return choice.message.content or ""

        # Append the assistant message (with tool_calls) to history
        messages.append(choice.message.model_dump())

        # Execute each tool call locally
        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            result = await _execute_tool(fn_name, fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # Safety: if we exhausted iterations, return whatever we have
    last_assistant = [m for m in messages if m.get("role") == "assistant"]
    if last_assistant:
        content = last_assistant[-1].get("content", "")
        if content:
            return content
    return "(Cena: max iterations reached without a final response)"


# =========================================================================
# Session management  (public API used by the server layer)
# =========================================================================

def create_session() -> str:
    """Create a new chat session and return its ID."""
    session = Session()
    _sessions[session.session_id] = session
    return session.session_id


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)


def delete_session(session_id: str) -> None:
    """Clean up a session and all associated state."""
    _sessions.pop(session_id, None)
    mapping_store.clear_session(session_id)
    _pending_images.pop(session_id, None)
    _last_sanitized.pop(session_id, None)


# =========================================================================
# Main entry point
# =========================================================================

async def process_message(
    session_id: str,
    text: str,
    model: str | None = None,
    image_bytes: bytes | None = None,
    mime_type: str | None = None,
) -> dict:
    """
    Run the full privacy pipeline for one user message.

    Nemotron (local, via vLLM) orchestrates:
      redact → transform → cloud relay → re-reference

    The cloud relay uses Claude Agent SDK (for Claude models) or OpenAI API
    (for GPT models) to perform the actual user task.

    Args:
        session_id:  Chat session UUID.
        text:        Raw user message (may contain PII).
        model:       Cloud LLM model to use (falls back to DEFAULT_CLOUD_MODEL).
        image_bytes: Optional raw image bytes from the user.
        mime_type:   Image MIME type if image_bytes is provided.

    Returns:
        dict with "response", "sanitized_prompt", "model", "had_image".
    """
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Unknown session: {session_id}")

    model = model or DEFAULT_CLOUD_MODEL
    has_image = bool(image_bytes and mime_type)

    # Stash the image where tools can reach it
    if has_image:
        _pending_images[session_id] = {
            "raw": image_bytes,
            "mime_type": mime_type,
            "transformed": None,
        }

    # Build the prompt Nemotron will reason over
    agent_prompt = (
        f"Process this user message through the privacy pipeline.\n\n"
        f"Metadata:\n"
        f"- Session ID: {session_id}\n"
        f"- Requested Cloud Model: {model}\n"
        f"- Has Image: {has_image}\n\n"
        f"User Message:\n{text}"
    )

    # Run the local Nemotron tool-calling loop
    final_response = await _run_nemotron_pipeline(agent_prompt)

    # Clean up per-message transient state
    _pending_images.pop(session_id, None)

    return {
        "response": final_response,
        "sanitized_prompt": _last_sanitized.get(session_id, text),
        "model": model,
        "had_image": has_image,
    }
