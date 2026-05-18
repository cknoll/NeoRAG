"""Structural validator: parse raw LLM output into the Answer schema.

Schema / JSON-decoding errors become :class:`Violation` objects of kind
``"structural"`` (improvement-plan2 step 3).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .schema import Answer
from .utils import _extract_json_object
from .violation import Violation


def validate_structural(answer_raw: str) -> Tuple[Optional[Answer], List[Violation]]:
    """Try to parse ``answer_raw`` into an :class:`Answer`.

    Returns ``(parsed_or_none, violations)``. On success, ``violations``
    is empty. On failure, ``parsed_or_none`` is ``None`` and the returned
    list contains exactly one ``Violation(kind="structural", ...)``.

    This intentionally duplicates a small amount of logic from
    :func:`neorag.generate.generate_structured_answer` so that the
    validator can also be applied to LLM output produced elsewhere
    (e.g. during refinement, where we receive a raw string and need to
    re-validate it from scratch).
    """
    import json

    from pydantic import ValidationError

    if not isinstance(answer_raw, str) or not answer_raw.strip():
        return None, [
            Violation(
                kind="structural",
                message="LLM response is empty.",
            )
        ]

    candidate = _extract_json_object(answer_raw)
    if candidate is None:
        return None, [
            Violation(
                kind="structural",
                message="No JSON object found in LLM response.",
            )
        ]

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        return None, [
            Violation(
                kind="structural",
                message=f"JSON decoding failed: {e}",
            )
        ]

    try:
        answer = Answer.model_validate(data)
    except ValidationError as e:
        return None, [
            Violation(
                kind="structural",
                message=f"Schema validation failed: {e}",
            )
        ]

    return answer, []
