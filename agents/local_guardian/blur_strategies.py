"""
Blur Strategies — Generalise PII values to preserve LLM utility while
reducing specificity enough to prevent identification.

Each PII type has a dedicated blur function that produces an *approximate*
value: useful for the cloud LLM to reason about, but not precise enough
to pinpoint the individual.

Examples:
  "February 6th, 2024"  →  "early February 2024"
  "$47,523.12"           →  "approximately $48,000"
  "123 Main St, Boston"  →  "an address in Boston"
  "32 years old"         →  "in their 30s"
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Per-type blur functions
# ---------------------------------------------------------------------------


def blur_date(value: str) -> str:
    """Blur a date to an approximate period (early/mid/late month)."""
    formats = [
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%B %d",
        "%b %d",
    ]

    # Strip ordinal suffixes (1st, 2nd, 3rd, 4th …)
    cleaned = re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", value).strip()

    parsed: datetime | None = None
    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            break
        except ValueError:
            continue

    if parsed is not None:
        day = parsed.day
        month = parsed.strftime("%B")
        period = "early" if day <= 10 else ("mid" if day <= 20 else "late")
        if parsed.year and parsed.year > 1:
            return f"{period} {month} {parsed.year}"
        return f"{period} {month}"

    # Fallback — can't parse, provide something vague
    return "around that time period"


def blur_money(value: str) -> str:
    """Blur a monetary amount to the nearest "nice" round number."""
    cleaned = re.sub(r"[,$\s]", "", value)
    try:
        amount = float(cleaned)
    except ValueError:
        return "a monetary amount"

    if amount < 100:
        rounded = round(amount / 10) * 10
    elif amount < 1_000:
        rounded = round(amount / 100) * 100
    elif amount < 10_000:
        rounded = round(amount / 1_000) * 1_000
    elif amount < 100_000:
        rounded = round(amount / 5_000) * 5_000
    elif amount < 1_000_000:
        rounded = round(amount / 50_000) * 50_000
    else:
        magnitude = 10 ** max(0, len(str(int(amount))) - 1)
        rounded = round(amount / magnitude) * magnitude

    return f"approximately ${rounded:,.0f}"


def blur_location(value: str) -> str:
    """Blur a location — full addresses to city-level, cities left intact."""
    # Full street address → city-level
    if re.search(
        r"\d+\s+\w+\s+(?:st|street|ave|avenue|rd|road|blvd|boulevard|"
        r"dr|drive|ln|lane|ct|court|way|pl|place|cir|circle|ter|terrace)",
        value,
        re.IGNORECASE,
    ):
        parts = [p.strip() for p in value.split(",")]
        if len(parts) >= 2:
            return f"an address in {parts[-2]}"
        return "a residential address"

    # Already city-level or broader — keep as-is
    return value


def blur_address(value: str) -> str:
    """Blur a street address to neighbourhood / city level."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) >= 2:
        return f"a location in {parts[-1]}"
    # Drop the street number and keep the street name
    no_number = re.sub(r"^\d+\s+", "", value)
    return f"near {no_number}" if no_number != value else "a street address"


def blur_age(value: str) -> str:
    """Blur an exact age to a decade range."""
    match = re.search(r"\d+", value)
    if match:
        age = int(match.group())
        decade = (age // 10) * 10
        return f"in their {decade}s"
    return "an adult"


def blur_organization(value: str) -> str:
    """Replace a specific organisation with a generic industry descriptor."""
    lower = value.lower()
    _industry_keywords: list[tuple[list[str], str]] = [
        (["tech", "software", "comput", "ai ", "digital", "cyber"], "a technology company"),
        (["bank", "capital", "financ", "invest", "insur"], "a financial institution"),
        (["health", "medic", "hospital", "pharma", "clinic", "bio"], "a healthcare organization"),
        (["univers", "college", "school", "academ", "institute"], "an educational institution"),
        (["law", "legal", "attorney"], "a legal firm"),
        (["gov", "feder", "state ", "depart"], "a government agency"),
    ]
    for keywords, descriptor in _industry_keywords:
        if any(k in lower for k in keywords):
            return descriptor
    return "an organization"


def blur_quantity(value: str) -> str:
    """Round a quantity to a coarse granularity."""
    match = re.search(r"[\d,.]+", value)
    if match:
        try:
            num = float(match.group().replace(",", ""))
            if num < 10:
                return f"approximately {round(num)}"
            if num < 100:
                return f"approximately {round(num / 5) * 5}"
            magnitude = 10 ** max(0, len(str(int(num))) - 2)
            rounded = round(num / magnitude) * magnitude
            return f"approximately {rounded:,.0f}"
        except ValueError:
            pass
    return "a quantity"


def blur_url(value: str) -> str:
    """Strip the URL path, keeping only the top-level domain."""
    match = re.match(r"https?://([^/]+)", value)
    if match:
        domain = match.group(1)
        parts = domain.split(".")
        if len(parts) >= 2:
            return f"a page on {parts[-2]}.{parts[-1]}"
    return "a web page"


def blur_zipcode(value: str) -> str:
    """Mask a ZIP code to the first 3 digits (covers ~40k people)."""
    digits = re.sub(r"[^0-9]", "", value)
    if len(digits) >= 3:
        return f"{digits[:3]}xx"
    return "a zip code area"


def blur_demographic(value: str) -> str:
    return "a demographic group"


def blur_event(value: str) -> str:
    return "an event"


def blur_work_of_art(value: str) -> str:
    return "a creative work"


def blur_legal_ref(value: str) -> str:
    return "a legal reference"


def blur_product(value: str) -> str:
    return "a product"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BLUR_FUNCTIONS: dict[str, Callable[[str], str]] = {
    "DATE": blur_date,
    "TIME": blur_date,
    "MONEY": blur_money,
    "LOCATION": blur_location,
    "ADDRESS": blur_address,
    "ORGANIZATION": blur_organization,
    "QUANTITY": blur_quantity,
    "DEMOGRAPHIC": blur_demographic,
    "URL": blur_url,
    "ZIPCODE": blur_zipcode,
    "AGE": blur_age,
    "EVENT": blur_event,
    "WORK_OF_ART": blur_work_of_art,
    "LEGAL_REF": blur_legal_ref,
    "PRODUCT": blur_product,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def blur(value: str, pii_type: str) -> Optional[str]:
    """
    Return a generalised version of *value* for the given *pii_type*.

    Returns ``None`` when no blur strategy is registered (caller should
    fall back to full redaction).
    """
    func = _BLUR_FUNCTIONS.get(pii_type)
    if func is not None:
        return func(value)
    return None
