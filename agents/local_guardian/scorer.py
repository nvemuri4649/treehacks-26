"""
Sensitivity & Utility Scorer — Assigns risk and usefulness scores to PII spans.

Scoring dimensions
==================
1. **Base sensitivity** — inherent risk of the PII *type* (SSN=1.0, date=0.40).
2. **Context sensitivity** — boost when surrounding text contains medical,
   financial, legal, or identity-related keywords.
3. **Co-occurrence risk** — boost when many distinct PII types appear together
   (quasi-identifier combinations increase re-identification probability).
4. **Utility** — how important the *exact* value is for the cloud LLM to
   produce a high-quality answer.

Action decision matrix
======================
+-------------------+-------------------+----------+
| Sensitivity       | Utility           | Action   |
+-------------------+-------------------+----------+
| ≥ 0.80 (critical) | any              | REDACT   |
| 0.50 – 0.79      | ≥ 0.45            | BLUR     |
| 0.50 – 0.79      | < 0.45            | REDACT   |
| 0.30 – 0.49      | ≥ 0.40            | BLUR     |
| 0.30 – 0.49      | < 0.40            | KEEP     |
| < 0.30           | any               | KEEP     |
+-------------------+-------------------+----------+
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.local_guardian.pii_detector import PIISpan


# ---------------------------------------------------------------------------
# Scored output
# ---------------------------------------------------------------------------


@dataclass
class ScoredSpan:
    """A PII span enriched with sensitivity, utility, and a recommended action."""

    span: "PIISpan"
    base_sensitivity: float
    context_boost: float
    co_occurrence_boost: float
    final_sensitivity: float
    utility: float
    action: str  # "KEEP" | "BLUR" | "REDACT"


# ---------------------------------------------------------------------------
# Base sensitivity look-up   (0.0 = benign, 1.0 = critical)
# ---------------------------------------------------------------------------

_BASE_SENSITIVITY: dict[str, float] = {
    # Critical — always redact
    "SSN": 1.00,
    "CREDIT_CARD": 1.00,
    "PASSPORT": 0.95,
    "DRIVER_LICENSE": 0.95,
    "BANK_ACCOUNT": 0.95,
    # Directly identifying
    "PERSON": 0.90,
    "EMAIL": 0.85,
    "PHONE": 0.85,
    "IP_ADDRESS": 0.80,
    "ADDRESS": 0.80,
    # Moderate
    "URL": 0.60,
    "DEMOGRAPHIC": 0.55,
    "ZIPCODE": 0.55,
    "LOCATION": 0.50,
    "AGE": 0.50,
    "MONEY": 0.45,
    "ORGANIZATION": 0.40,
    "DATE": 0.40,
    # Low
    "LEGAL_REF": 0.35,
    "TIME": 0.25,
    "EVENT": 0.20,
    "QUANTITY": 0.15,
    "WORK_OF_ART": 0.10,
    "PRODUCT": 0.10,
}


# ---------------------------------------------------------------------------
# Context patterns that boost sensitivity
# ---------------------------------------------------------------------------

_CONTEXT_DOMAINS: list[dict] = [
    {
        "name": "medical",
        "pattern": re.compile(
            r"\b(?:diagnos\w+|prescri\w+|symptom|treatment|hospital|doctor"
            r"|dr\.|patient|medical|health|illness|diseas\w+|medication"
            r"|surgery|clinic|therap\w+|condition|blood\s*(?:type|pressure)"
            r"|allerg\w+|insur(?:ance|ed))\b",
            re.IGNORECASE,
        ),
        "boost": 0.25,
        "window": 120,
    },
    {
        "name": "financial",
        "pattern": re.compile(
            r"\b(?:account|balance|salary|income|tax|bank|loan|credit"
            r"|debit|payment|mortgage|invest\w+|portfolio|dividend"
            r"|interest\s*rate|routing\s*number|net\s*worth|fico)\b",
            re.IGNORECASE,
        ),
        "boost": 0.20,
        "window": 120,
    },
    {
        "name": "legal",
        "pattern": re.compile(
            r"\b(?:case\s*(?:number|no\.?|#)|defendant|plaintiff|attorney"
            r"|lawyer|court|lawsuit|charge[sd]?|verdict|sentence[d]?"
            r"|probation|arrest\w*|felony|misdemeanor)\b",
            re.IGNORECASE,
        ),
        "boost": 0.20,
        "window": 100,
    },
    {
        "name": "identity",
        "pattern": re.compile(
            r"\b(?:born|birth\s*(?:day|date)|age[d:]|gender|sex\b|race"
            r"|ethnicit\w+|nationalit\w+|religion|orientation|married"
            r"|divorced|maiden\s*name|citizen\w*)\b",
            re.IGNORECASE,
        ),
        "boost": 0.15,
        "window": 80,
    },
]


# ---------------------------------------------------------------------------
# Utility look-up   (0.0 = not needed at all, 1.0 = essential for quality)
# ---------------------------------------------------------------------------

_BASE_UTILITY: dict[str, float] = {
    # Zero utility — hiding never hurts answer quality
    "SSN": 0.00,
    "CREDIT_CARD": 0.00,
    "PASSPORT": 0.00,
    "DRIVER_LICENSE": 0.00,
    "BANK_ACCOUNT": 0.00,
    "PHONE": 0.05,
    # Low utility
    "IP_ADDRESS": 0.10,
    "EMAIL": 0.15,
    "ZIPCODE": 0.15,
    "ADDRESS": 0.20,
    "DEMOGRAPHIC": 0.25,
    # Moderate — blurring preserves enough
    "PERSON": 0.30,
    "URL": 0.30,
    "LEGAL_REF": 0.40,
    "AGE": 0.45,
    "ORGANIZATION": 0.50,
    "TIME": 0.50,
    "LOCATION": 0.55,
    "MONEY": 0.55,
    "EVENT": 0.55,
    "DATE": 0.60,
    # High utility — important for output quality
    "PRODUCT": 0.60,
    "WORK_OF_ART": 0.60,
    "QUANTITY": 0.65,
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _get_context_window(text: str, start: int, end: int, window: int) -> str:
    """Return the substring of *text* within *window* chars of the span."""
    return text[max(0, start - window) : min(len(text), end + window)]


def _context_boost(span: "PIISpan", text: str) -> float:
    """
    Compute an additive sensitivity boost from surrounding context.

    For each domain (medical, financial, legal, identity), if keywords are
    found near the span the domain's boost is applied.  Only the maximum
    single-domain boost is returned (they don't stack linearly — we take
    the max to avoid over-boosting).
    """
    best = 0.0
    for domain in _CONTEXT_DOMAINS:
        ctx = _get_context_window(text, span.start, span.end, domain["window"])
        if domain["pattern"].search(ctx):
            best = max(best, domain["boost"])
    return best


def _co_occurrence_boost(all_spans: list["PIISpan"]) -> float:
    """
    Boost all spans when many distinct PII types co-occur, since
    combinations of quasi-identifiers dramatically increase re-identification
    probability (Sweeney, 2000).
    """
    unique_types = {s.pii_type for s in all_spans}
    n = len(unique_types)
    if n <= 1:
        return 0.0
    if n <= 3:
        return 0.05
    if n <= 5:
        return 0.10
    return 0.15


def _decide_action(sensitivity: float, utility: float) -> str:
    """Map (sensitivity, utility) to an action string."""
    if sensitivity >= 0.80:
        return "REDACT"
    if sensitivity >= 0.50:
        return "BLUR" if utility >= 0.45 else "REDACT"
    if sensitivity >= 0.30:
        return "BLUR" if utility >= 0.40 else "KEEP"
    return "KEEP"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_spans(spans: list["PIISpan"], text: str) -> list[ScoredSpan]:
    """
    Score every detected PII span and assign a recommended action.

    Args:
        spans: Non-overlapping ``PIISpan`` list from the detector.
        text:  The full original text (used for context window analysis).

    Returns:
        A list of ``ScoredSpan`` objects in the same order as *spans*.
    """
    co_boost = _co_occurrence_boost(spans)

    scored: list[ScoredSpan] = []
    for span in spans:
        base_sens = _BASE_SENSITIVITY.get(span.pii_type, 0.50)
        ctx_boost = _context_boost(span, text)
        final_sens = min(1.0, base_sens + ctx_boost + co_boost)
        utility = _BASE_UTILITY.get(span.pii_type, 0.30)
        action = _decide_action(final_sens, utility)

        scored.append(
            ScoredSpan(
                span=span,
                base_sensitivity=base_sens,
                context_boost=ctx_boost,
                co_occurrence_boost=co_boost,
                final_sensitivity=final_sens,
                utility=utility,
                action=action,
            )
        )

    return scored
