"""Violation dataclass shared by all validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Violation:
    """A single constraint violation reported by a validator.

    Attributes
    ----------
    kind:
        Short tag identifying the validator that produced this violation,
        e.g. ``"structural"``, ``"groundedness"`` or ``"shacl"``.
    message:
        Human-readable description of what went wrong.
    location:
        Free-form string pointing into the answer for a reviewer's
        convenience (e.g. ``"claims[0].citations[1]"``). May be ``None``
        when the violation applies to the answer as a whole. Kept as a
        plain string for v0.1 -- a structured location type is P2.
    """

    kind: str
    message: str
    location: Optional[str] = None
