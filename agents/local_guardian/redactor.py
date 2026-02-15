"""
Redactor — Detects, scores, and replaces PII in user prompts.

Full pipeline (all local, nothing leaves the machine):

  1. **DETECT**     — Find PII spans via regex patterns + spaCy NER.
  2. **SCORE**      — Assign sensitivity & utility scores per span,
                      with context-aware boosts (medical, financial, …).
  3. **ADVERSARIAL** — Run a re-identification risk check; escalate
                      any BLUR decisions to REDACT when quasi-identifier
                      combinations make inference feasible.
  4. **APPLY**      — Replace spans in the text:
                        • REDACT → deterministic token ``[TYPE_N]``
                        • BLUR   → token ``[TYPE_N]`` + a context hint
                                   appended to the sanitized text
                        • KEEP   → left unchanged
  5. **REPORT**     — Generate a ``PrivacyReport`` for the UI.

Function signature and return contract are stable — callers receive
``(sanitized_text, new_mapping)`` as before.  The privacy report is
stored internally and retrieved via ``get_last_report(session_id)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from agents.local_guardian.mapping_store import store as mapping_store
from agents.local_guardian.pii_detector import detect, PIISpan
from agents.local_guardian.scorer import score_spans, ScoredSpan
from agents.local_guardian.blur_strategies import blur
from agents.local_guardian.adversarial_analyzer import analyze as adversarial_analyze


# ---------------------------------------------------------------------------
# Privacy report model
# ---------------------------------------------------------------------------


@dataclass
class EntityReport:
    """One entity's summary for the UI."""

    pii_type: str
    action: str  # KEEP | BLUR | REDACT
    token: str | None = None
    hint: str | None = None  # blurred approximation, if applicable


@dataclass
class PrivacyReport:
    """Full privacy report for one redaction pass."""

    total_detected: int = 0
    redacted_count: int = 0
    blurred_count: int = 0
    kept_count: int = 0
    entities: list[EntityReport] = field(default_factory=list)
    adversarial_risk: float = 0.0
    privacy_confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    spacy_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "total_detected": self.total_detected,
            "redacted_count": self.redacted_count,
            "blurred_count": self.blurred_count,
            "kept_count": self.kept_count,
            "entities": [
                {k: v for k, v in asdict(e).items() if v is not None}
                for e in self.entities
            ],
            "adversarial_risk": self.adversarial_risk,
            "privacy_confidence": self.privacy_confidence,
            "warnings": self.warnings,
            "spacy_active": self.spacy_active,
        }


# ---------------------------------------------------------------------------
# Per-session report cache
# ---------------------------------------------------------------------------

_last_reports: dict[str, PrivacyReport] = {}


def get_last_report(session_id: str) -> PrivacyReport | None:
    """Retrieve the most recent privacy report for a session."""
    return _last_reports.get(session_id)


def clear_report(session_id: str) -> None:
    """Discard the cached report for a session."""
    _last_reports.pop(session_id, None)


# ---------------------------------------------------------------------------
# Core redaction logic
# ---------------------------------------------------------------------------


def redact(text: str, session_id: str) -> tuple[str, dict[str, str]]:
    """
    Scan *text* for personal information, replace each occurrence with a
    deterministic token, and return the sanitised text plus the mapping.

    Args:
        text:       Raw user input (may contain PII).
        session_id: Current chat session UUID.

    Returns:
        A 2-tuple:
          sanitized_text  — the input with PII replaced by tokens like
                            ``[PERSON_1]``, plus context hints for blurred
                            items appended at the end.
          new_mapping     — dict of **newly created** ``{token: original_value}``
                            pairs produced by this call (subset of the full
                            session mapping).

    Contract:
        * Tokens use the format ``[TYPE_N]`` (e.g. ``[PERSON_1]``).
        * The same original value in the same session always produces the
          same token (idempotent, via the MappingStore).
        * ``mapping_store.add()`` is called for every redacted or blurred span.
    """
    from agents.local_guardian.pii_detector import is_spacy_available

    # ── 1. DETECT ─────────────────────────────────────────────────────
    spans: list[PIISpan] = detect(text)

    if not spans:
        _last_reports[session_id] = PrivacyReport(
            spacy_active=is_spacy_available(),
            privacy_confidence=1.0,
        )
        return text, {}

    # ── 2. SCORE ──────────────────────────────────────────────────────
    scored: list[ScoredSpan] = score_spans(spans, text)

    # ── 3. ADVERSARIAL ANALYSIS ───────────────────────────────────────
    adv_report = adversarial_analyze(scored, text)

    # Escalate spans that the adversarial check flagged
    for idx in adv_report.escalated_spans:
        if idx < len(scored):
            old = scored[idx]
            scored[idx] = ScoredSpan(
                span=old.span,
                base_sensitivity=old.base_sensitivity,
                context_boost=old.context_boost,
                co_occurrence_boost=old.co_occurrence_boost,
                final_sensitivity=old.final_sensitivity,
                utility=old.utility,
                action="REDACT",
            )

    # ── 4. APPLY REPLACEMENTS ─────────────────────────────────────────
    new_mapping: dict[str, str] = {}
    blur_hints: list[str] = []
    entity_reports: list[EntityReport] = []

    redacted_count = 0
    blurred_count = 0
    kept_count = 0

    # Process spans in reverse order to preserve character offsets
    sorted_scored = sorted(scored, key=lambda s: s.span.start, reverse=True)
    sanitized = text

    for ss in sorted_scored:
        span = ss.span
        original = span.text

        if ss.action == "KEEP":
            kept_count += 1
            entity_reports.append(EntityReport(pii_type=span.pii_type, action="KEEP"))
            continue

        # Both REDACT and BLUR get a token
        token = mapping_store.add(session_id, span.pii_type, original)
        sanitized = sanitized[: span.start] + token + sanitized[span.end :]
        new_mapping[token] = original

        if ss.action == "BLUR":
            blurred_count += 1
            blurred_value = blur(original, span.pii_type)
            if blurred_value:
                blur_hints.append(f"  {token} ≈ {blurred_value}")
                entity_reports.append(
                    EntityReport(
                        pii_type=span.pii_type,
                        action="BLUR",
                        token=token,
                        hint=blurred_value,
                    )
                )
            else:
                # No blur strategy — fall through to REDACT presentation
                redacted_count += 1
                entity_reports.append(
                    EntityReport(pii_type=span.pii_type, action="REDACT", token=token)
                )
        else:
            # REDACT
            redacted_count += 1
            entity_reports.append(
                EntityReport(pii_type=span.pii_type, action="REDACT", token=token)
            )

    # ── 5. CONTEXT HINTS (kept local only — NEVER sent to cloud) ─────
    # blur_hints are stored in EntityReport for the UI privacy panel
    # but are NOT appended to the sanitized text.

    # ── 6. BUILD PRIVACY REPORT ───────────────────────────────────────
    # Reverse entity_reports so they're in document order (we built them
    # backwards because of reverse offset processing)
    entity_reports.reverse()

    report = PrivacyReport(
        total_detected=len(scored),
        redacted_count=redacted_count,
        blurred_count=blurred_count,
        kept_count=kept_count,
        entities=entity_reports,
        adversarial_risk=adv_report.risk_score,
        privacy_confidence=adv_report.privacy_confidence,
        warnings=adv_report.warnings,
        spacy_active=is_spacy_available(),
    )
    _last_reports[session_id] = report

    return sanitized, new_mapping
