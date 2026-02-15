"""
Adversarial Re-identification Analyzer
=======================================
Estimates the risk that an attacker could re-identify the user from the
*sanitized* text alone, using techniques analogous to quasi-identifier
analysis (Sweeney, 2000).

Checks performed
----------------
1. **Quasi-identifier combinations** — certain PII type combos (e.g.
   age + zip + gender) uniquely identify individuals even when each
   value alone is "blurred".
2. **Directly-identifying singletons** — some types (PERSON, EMAIL,
   PHONE, SSN) are never safe to merely blur; they must be fully
   redacted.
3. **Information density** — when a high proportion of detected PII
   remains partially exposed, cumulative inference risk grows.

Output
------
An ``AdversarialReport`` with:
  * ``risk_score``       — 0.0 (safe) … 1.0 (high risk)
  * ``escalated_spans``  — indices that should be promoted BLUR → REDACT
  * ``warnings``         — human-readable explanation strings
  * ``privacy_confidence`` — 1 − risk_score, for the UI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.local_guardian.scorer import ScoredSpan


@dataclass
class AdversarialReport:
    """Results of the adversarial re-identification analysis."""

    risk_score: float = 0.0
    escalated_spans: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def privacy_confidence(self) -> float:
        """User-facing confidence that privacy is preserved (0–1)."""
        return round(1.0 - self.risk_score, 3)


# ---------------------------------------------------------------------------
# Quasi-identifier type sets
# ---------------------------------------------------------------------------

_QUASI_ID_COMBOS: list[frozenset[str]] = [
    frozenset({"DATE", "LOCATION", "DEMOGRAPHIC"}),
    frozenset({"DATE", "ZIPCODE", "DEMOGRAPHIC"}),
    frozenset({"AGE", "LOCATION", "ORGANIZATION"}),
    frozenset({"AGE", "ZIPCODE", "DEMOGRAPHIC"}),
    frozenset({"PERSON", "ORGANIZATION"}),
    frozenset({"PERSON", "LOCATION"}),
    frozenset({"DATE", "ORGANIZATION", "LOCATION"}),
    frozenset({"PERSON", "DATE", "LOCATION"}),
]

# Types that are directly identifying — BLUR is never safe
_DIRECT_IDENTIFIERS: frozenset[str] = frozenset({
    "PERSON",
    "EMAIL",
    "PHONE",
    "SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "PASSPORT",
    "DRIVER_LICENSE",
    "BANK_ACCOUNT",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(
    scored_spans: list["ScoredSpan"],
    sanitized_text: str,
) -> AdversarialReport:
    """
    Analyse the scored spans for re-identification risk and return an
    ``AdversarialReport``.

    Any span indices listed in ``escalated_spans`` should be promoted from
    BLUR to REDACT by the caller.
    """
    warnings: list[str] = []
    escalated: set[int] = set()
    risk_components: list[float] = []

    # ── 1. Check directly-identifying singletons ─────────────────────
    for i, ss in enumerate(scored_spans):
        if ss.span.pii_type in _DIRECT_IDENTIFIERS and ss.action == "BLUR":
            risk_components.append(0.40)
            escalated.add(i)
            warnings.append(
                f"{ss.span.pii_type} is a direct identifier and must not be "
                f"merely blurred — escalating to REDACT."
            )

    # ── 2. Quasi-identifier combination check ────────────────────────
    exposed_types: set[str] = set()
    exposed_type_indices: dict[str, list[int]] = {}
    for i, ss in enumerate(scored_spans):
        if ss.action in ("BLUR", "KEEP"):
            exposed_types.add(ss.span.pii_type)
            exposed_type_indices.setdefault(ss.span.pii_type, []).append(i)

    for combo in _QUASI_ID_COMBOS:
        if combo.issubset(exposed_types):
            combo_risk = 0.25 + 0.05 * len(combo)
            risk_components.append(combo_risk)
            combo_str = ", ".join(sorted(combo))
            warnings.append(
                f"Quasi-identifier combination [{combo_str}] detected — "
                f"together these can narrow re-identification significantly."
            )
            # Escalate the BLUR members of this combo
            for pii_type in combo:
                for idx in exposed_type_indices.get(pii_type, []):
                    if scored_spans[idx].action == "BLUR":
                        escalated.add(idx)

    # ── 3. Information density check ─────────────────────────────────
    total = len(scored_spans)
    exposed_count = sum(1 for ss in scored_spans if ss.action in ("BLUR", "KEEP"))

    if total >= 4 and exposed_count / total > 0.60:
        density_risk = 0.15 * (exposed_count / total)
        risk_components.append(density_risk)
        warnings.append(
            f"High information density: {exposed_count}/{total} entities "
            f"remain partially exposed — cumulative inference risk is elevated."
        )

    # ── 4. Compute overall risk score ────────────────────────────────
    if risk_components:
        # Combine probabilities: P(A∪B) = 1 − Π(1−pᵢ)
        product = 1.0
        for r in risk_components:
            product *= 1.0 - min(r, 1.0)
        overall_risk = 1.0 - product
    else:
        overall_risk = 0.0

    return AdversarialReport(
        risk_score=round(min(overall_risk, 1.0), 3),
        escalated_spans=sorted(escalated),
        warnings=warnings,
    )
