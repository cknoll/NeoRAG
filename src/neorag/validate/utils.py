"""Shared helpers for the NeoRAG validate package."""

from __future__ import annotations

import re
from typing import Optional

# Matches a ```json ... ``` or bare ``` ... ``` fenced block; some LLMs
# wrap JSON in markdown despite instructions not to.
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_json_object(text: str) -> Optional[str]:
    """Best-effort extraction of a single JSON object string from ``text``.

    Returns the substring (still as text, not parsed) or ``None`` if no
    plausible JSON object is found. Intentionally permissive: the
    structural validator turns parse failures into violations, so it is
    fine for this helper to occasionally return malformed candidates.
    """
    if not text:
        return None

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    m = _FENCED_JSON_RE.search(text)
    if m is not None:
        return m.group(1)

    # Fall back to the substring between the first '{' and last '}'.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None
