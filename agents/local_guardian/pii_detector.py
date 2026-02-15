"""
PII Detector — Multi-layer detection of personally identifiable information.

Combines two complementary strategies:
  1. **Regex patterns** for structured PII (SSN, email, phone, credit card,
     IP address, dates, URLs, zip codes, etc.).
  2. **spaCy NER** (optional — gracefully degraded when not installed) for
     named entities (PERSON, ORG, GPE, LOC, DATE, MONEY, NORP, etc.).

Overlapping spans are resolved by preferring higher-confidence, more-specific
detections and longer spans.

All processing is local — nothing leaves the machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


#---------------------------------------------------------------------------
#Optional spaCy import — enhanced NER when available, regex-only fallback
#---------------------------------------------------------------------------

try:
    import spacy

    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError:
        _nlp = None
    _SPACY_AVAILABLE = _nlp is not None
except ImportError:
    _nlp = None
    _SPACY_AVAILABLE = False


#---------------------------------------------------------------------------
#Data model
#---------------------------------------------------------------------------


@dataclass
class PIISpan:
    """A single detected PII span within text."""

    text: str
    start: int
    end: int
    pii_type: str
    detection_method: str  # "regex" | "ner"
    confidence: float = 1.0


#---------------------------------------------------------------------------
#Regex patterns — (pii_type, compiled_pattern)
#Ordered from most to least specific so overlap resolution favours precision.
#---------------------------------------------------------------------------

_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    #── Critical identifiers ──────────────────────────────────────────
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ),
    #── Contact information ───────────────────────────────────────────
    (
        "EMAIL",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ),
    ),
    (
        "PHONE",
        re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
    ),
    #── Network / Digital ─────────────────────────────────────────────
    (
        "IP_ADDRESS",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
    (
        "URL",
        re.compile(r"https?://[^\s<>\"']+"),
    ),
    #── Dates (multiple formats) ──────────────────────────────────────
    (
        "DATE",
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DATE",
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DATE",
        re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    ),
    (
        "DATE",
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    ),
    #── Geographic ────────────────────────────────────────────────────
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,6}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*"
            r"\s+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|"
            r"Lane|Ln|Court|Ct|Way|Place|Pl|Circle|Cir|Terrace|Ter)"
            r"\.?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ZIPCODE",
        re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    ),
    #── Monetary amounts ───────────────────────────────────────────────
    (
        "MONEY",
        re.compile(
            r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"
            r"|\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:dollars?|USD)\b",
            re.IGNORECASE,
        ),
    ),
    #── Age expressions ───────────────────────────────────────────────
    (
        "AGE",
        re.compile(
            r"\b(\d{1,3})\s*[-–]?\s*(?:years?\s*old|yr\.?s?\s*old|y/?o)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "AGE",
        re.compile(r"\bage[d:]\s*(\d{1,3})\b", re.IGNORECASE),
    ),
]

#---------------------------------------------------------------------------
#spaCy entity label → our PII type
#---------------------------------------------------------------------------

_SPACY_TYPE_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "DATE": "DATE",
    "TIME": "TIME",
    "MONEY": "MONEY",
    "NORP": "DEMOGRAPHIC",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "LAW": "LEGAL_REF",
    "PRODUCT": "PRODUCT",
    "QUANTITY": "QUANTITY",
}


#---------------------------------------------------------------------------
#Detection engines
#---------------------------------------------------------------------------


def _detect_regex(text: str) -> list[PIISpan]:
    """Run all compiled regex patterns over *text*."""
    spans: list[PIISpan] = []
    for pii_type, pattern in _REGEX_PATTERNS:
        for match in pattern.finditer(text):
            spans.append(
                PIISpan(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    pii_type=pii_type,
                    detection_method="regex",
                    confidence=0.95,
                )
            )
    return spans


_NER_STOPWORDS: frozenset[str] = frozenset({
    "ssn", "dob", "dob:", "id", "id:", "ph", "tel", "fax", "re:", "re",
    "cc", "bcc", "attn", "attn:", "n/a", "tbd", "asap", "eta", "etc",
    "mr", "mrs", "ms", "dr", "jr", "sr", "ii", "iii", "iv",
})


def _detect_ner(text: str) -> list[PIISpan]:
    """Run spaCy NER over *text* (no-op if spaCy is unavailable)."""
    if not _SPACY_AVAILABLE or _nlp is None:
        return []

    doc = _nlp(text)
    spans: list[PIISpan] = []
    for ent in doc.ents:
        pii_type = _SPACY_TYPE_MAP.get(ent.label_)
        if pii_type is None:
            continue
        #Skip common abbreviations / labels that NER picks up spuriously
        if ent.text.strip().lower() in _NER_STOPWORDS:
            continue
        #Skip very short entities (likely noise)
        if len(ent.text.strip()) < 2:
            continue
        spans.append(
            PIISpan(
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                pii_type=pii_type,
                detection_method="ner",
                confidence=0.85,
            )
        )
    return spans


#---------------------------------------------------------------------------
#Overlap resolution
#---------------------------------------------------------------------------


def _resolve_overlaps(spans: list[PIISpan]) -> list[PIISpan]:
    """
    Produce a list of non-overlapping spans by greedily keeping the
    highest-confidence / longest detection at each position.

    Preference order (when spans overlap):
      1. Higher confidence
      2. Longer span
      3. Earlier start position (tie-break)
    """
    if not spans:
        return []

    #Sort: highest confidence first, then longest, then earliest start
    ranked = sorted(
        spans,
        key=lambda s: (-s.confidence, -(s.end - s.start), s.start),
    )

    accepted: list[PIISpan] = []
    for candidate in ranked:
        overlaps = any(
            not (candidate.end <= a.start or candidate.start >= a.end)
            for a in accepted
        )
        if not overlaps:
            accepted.append(candidate)

    return sorted(accepted, key=lambda s: s.start)


#---------------------------------------------------------------------------
#Public API
#---------------------------------------------------------------------------


def detect(text: str) -> list[PIISpan]:
    """
    Detect all PII spans in *text* using regex patterns and (optionally)
    spaCy NER.  Returns a list of non-overlapping ``PIISpan`` objects
    sorted by position.
    """
    regex_spans = _detect_regex(text)
    ner_spans = _detect_ner(text)
    all_spans = regex_spans + ner_spans
    return _resolve_overlaps(all_spans)


def is_spacy_available() -> bool:
    """Check whether spaCy NER is loaded."""
    return _SPACY_AVAILABLE
