"""
Re-Referencing Engine — Restores original PII values in cloud LLM responses.

After the Cloud Relay returns a response that still contains redaction tokens
(e.g. [PERSON_1], [ADDRESS_1]), this module substitutes them back to the
real values using the session's mapping.

It also strips any residual "[Context — …]" hint blocks that the redactor
appended for the cloud LLM's benefit.
"""

from __future__ import annotations

import re


#Regex to match the context-hints block appended by the redactor.
#Handles the full section from header through all hint lines.
_CONTEXT_BLOCK_RE = re.compile(
    r"\n*\[Context\s*[—–-]\s*approximate values for reference.*?\]"
    r"(?:\n\s*\[.*?\].*)*",
    re.DOTALL,
)


def re_reference(text: str, mapping: dict[str, str]) -> str:
    """
    Replace every redaction token in *text* with its original PII value.

    Args:
        text:    The cloud LLM response containing tokens like ``[PERSON_1]``.
        mapping: ``{token: original_value}`` dict from the MappingStore.

    Returns:
        The response with all recognised tokens restored to real values and
        any residual context-hint blocks stripped.

    Notes:
        * Tokens are replaced longest-first to avoid substring collisions
          (e.g. ``[PERSON_10]`` must be replaced before ``[PERSON_1]``).
        * Tokens that appear inside markdown code fences are still replaced
          because the user likely wants to see the real values everywhere.
        * Any "[Context — approximate values …]" section that leaked into
          the LLM's reply is silently removed.
    """
    if not text:
        return text

    #1. Strip any residual context-hint blocks
    result = _CONTEXT_BLOCK_RE.sub("", text)

    #2. Replace tokens → original values
    if mapping:
        #Sort tokens longest-first so [PERSON_10] beats [PERSON_1]
        sorted_tokens = sorted(mapping.keys(), key=len, reverse=True)
        for token in sorted_tokens:
            result = result.replace(token, mapping[token])

    return result.strip()
