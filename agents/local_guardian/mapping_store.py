"""
MappingStore — Per-session storage for PII token mappings.

Keeps a bidirectional mapping between redaction tokens (e.g. [PERSON_1])
and the original PII values they replaced.  All data lives in-memory only
and is never persisted to disk or sent over the network.
"""

from __future__ import annotations

import threading
from collections import defaultdict


class MappingStore:
    """Thread-safe, session-scoped store for token <-> PII mappings."""

    def __init__(self) -> None:
        # {session_id: {token: original_value}}
        self._store: dict[str, dict[str, str]] = defaultdict(dict)
        # {session_id: {original_value: token}}  (reverse index for reuse)
        self._reverse: dict[str, dict[str, str]] = defaultdict(dict)
        # {session_id: {pii_type: counter}}
        self._counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, session_id: str, pii_type: str, original_value: str) -> str:
        """
        Register a PII value and return its token.

        If the same original_value was already registered in this session,
        the existing token is returned (idempotent).

        Args:
            session_id:     Current chat session UUID.
            pii_type:       Category string, e.g. "PERSON", "ADDRESS", "EMAIL".
            original_value: The raw PII text to redact.

        Returns:
            Token string, e.g. "[PERSON_1]".
        """
        with self._lock:
            # Reuse existing token for the same value in this session
            existing = self._reverse[session_id].get(original_value)
            if existing is not None:
                return existing

            # Mint a new token
            self._counters[session_id][pii_type] += 1
            counter = self._counters[session_id][pii_type]
            token = f"[{pii_type}_{counter}]"

            self._store[session_id][token] = original_value
            self._reverse[session_id][original_value] = token
            return token

    def get_mapping(self, session_id: str) -> dict[str, str]:
        """
        Return the full {token: original_value} mapping for a session.

        Returns an empty dict if the session has no mappings.
        """
        with self._lock:
            return dict(self._store.get(session_id, {}))

    def get_reverse_mapping(self, session_id: str) -> dict[str, str]:
        """
        Return {original_value: token} for a session (useful for the redactor).
        """
        with self._lock:
            return dict(self._reverse.get(session_id, {}))

    def clear_session(self, session_id: str) -> None:
        """Remove all mappings for a session."""
        with self._lock:
            self._store.pop(session_id, None)
            self._reverse.pop(session_id, None)
            self._counters.pop(session_id, None)


# Module-level singleton so all components share one store.
store = MappingStore()
