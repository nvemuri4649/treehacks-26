"""
Re-Referencing Engine — Restores original PII values in cloud LLM responses.

After the Cloud Relay returns a response that still contains redaction tokens
(e.g. [PERSON_1], [ADDRESS_1]), this module substitutes them back to the
real values using the session's mapping.
"""

from __future__ import annotations


def re_reference(text: str, mapping: dict[str, str]) -> str:
    """
    Replace every redaction token in *text* with its original PII value.

    Args:
        text:    The cloud LLM response containing tokens like ``[PERSON_1]``.
        mapping: ``{token: original_value}`` dict from the MappingStore.

    Returns:
        The response with all recognised tokens restored to real values.

    Notes:
        * Tokens are replaced longest-first to avoid substring collisions
          (e.g. ``[PERSON_10]`` must be replaced before ``[PERSON_1]``).
        * Tokens that appear inside markdown code fences are still replaced
          because the user likely wants to see the real values everywhere.
    """
    if not mapping:
        return text

    # Sort tokens longest-first so [PERSON_10] is replaced before [PERSON_1]
    sorted_tokens = sorted(mapping.keys(), key=len, reverse=True)

    result = text
    for token in sorted_tokens:
        result = result.replace(token, mapping[token])

    return result
