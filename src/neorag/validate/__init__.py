"""Neurosymbolic answer validation for NeoRAG.

This package hosts the Pydantic answer schema (``schema.py``) plus the
structural / groundedness / SHACL validators (improvement-plan2 step 3).

Public API:

- :func:`validate` -- top-level entry point that runs all three
  validators (structural, groundedness, SHACL) and returns the union
  of their :class:`Violation`\\ s.
- :class:`Violation` -- the dataclass returned by every validator.
- :class:`Answer`, :class:`Claim`, :class:`Citation` -- the Pydantic
  answer schema.
"""

from __future__ import annotations

from typing import Any, Iterable, List

from .groundedness import validate_groundedness
from .schema import Answer, Citation, Claim
from .shacl import validate_shacl
from .structural import validate_structural
from .violation import Violation

__all__ = [
    "Answer",
    "Citation",
    "Claim",
    "Violation",
    "validate",
    "validate_structural",
    "validate_groundedness",
    "validate_shacl",
]


def validate(
    answer_raw: str,
    retrieved_nodes: Iterable[Any],
) -> List[Violation]:
    """Run all validators on ``answer_raw`` and return their violations.

    The structural validator runs first; if it fails, the answer cannot
    be parsed and the downstream validators have nothing to operate on,
    so we return early with just the structural violation(s).
    Otherwise, groundedness and SHACL are both run and their violations
    are concatenated.
    """
    # Materialise once so the iterable can be consumed by both validators.
    nodes = list(retrieved_nodes)

    answer, structural_violations = validate_structural(answer_raw)
    if answer is None:
        return list(structural_violations)

    violations: List[Violation] = []
    violations.extend(structural_violations)
    violations.extend(validate_groundedness(answer, nodes))
    violations.extend(validate_shacl(answer, nodes))
    return violations
