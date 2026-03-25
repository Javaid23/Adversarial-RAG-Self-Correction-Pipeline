from __future__ import annotations


def citation_coverage(answer: str) -> float:
    """Toy metric: ratio of citation-like brackets in output."""
    if not answer:
        return 0.0
    cites = answer.count("[")
    return min(1.0, cites / 3)


def contains_uncertainty_when_needed(answer: str) -> bool:
    lower = answer.lower()
    return (
        "insufficient" in lower
        or "not enough" in lower
        or "cannot determine" in lower
        or "unclear" in lower
    )
