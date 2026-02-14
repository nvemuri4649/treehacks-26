"""
Redactor — Detects and replaces PII in user prompts with tokens.

=== PLACEHOLDER IMPLEMENTATION ===
This module exposes a stable interface that the rest of the system depends on.
Replace the body of `redact()` with your real PII-detection algorithm when ready.
The function signature and return contract must stay the same.
"""

from __future__ import annotations

from agents.local_guardian.mapping_store import store as mapping_store


def redact(text: str, session_id: str) -> tuple[str, dict[str, str]]:
    """
    Scan *text* for personal information, replace each occurrence with a
    deterministic token, and return the sanitised text plus the mapping.

    Args:
        text:       Raw user input (may contain PII).
        session_id: Current chat session UUID.

    Returns:
        A 2-tuple:
          sanitized_text  — the input with PII replaced by tokens like [PERSON_1].
          new_mapping     — dict of **newly created** {token: original_value} pairs
                            produced by this call (subset of the full session mapping).

    Contract:
        * Tokens MUST use the format ``[TYPE_N]`` where TYPE is an uppercase
          PII category (PERSON, ADDRESS, EMAIL, PHONE, SSN, …) and N is a
          per-session incrementing counter.
        * The same original value in the same session MUST always map to the
          same token (idempotent).
        * The function MUST call ``mapping_store.add(session_id, pii_type, value)``
          for every detected PII span so that the MappingStore stays in sync.

    Example (once implemented):
        >>> redact("Email john@acme.com about 123 Main St", "sess-1")
        ("Email [EMAIL_1] about [ADDRESS_1]",
         {"[EMAIL_1]": "john@acme.com", "[ADDRESS_1]": "123 Main St"})
    """
    # ------------------------------------------------------------------
    # TODO: Replace this block with the real detection / redaction logic.
    #
    # Typical approach:
    #   1. Run NER / regex / custom model over `text` to find PII spans.
    #   2. For each span, call:
    #        token = mapping_store.add(session_id, pii_type, span_text)
    #   3. Replace the span in `text` with the token.
    #   4. Collect {token: span_text} pairs into `new_mapping`.
    # ------------------------------------------------------------------

    sanitized_text = text
    new_mapping: dict[str, str] = {}

    return sanitized_text, new_mapping
