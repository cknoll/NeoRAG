"""Pydantic answer schema for NeoRAG (improvement-plan2 step 2).

The LLM is instructed (see :mod:`neorag.generate`) to emit JSON
matching :class:`Answer`. Each :class:`Claim` carries a textual
statement plus one or more :class:`Citation` objects pointing to a
specific chunk by ``(doc_id, chunk_idx_in_doc)`` -- chunk-level only
for v0.1 (sub-chunk byte spans are deferred to P2).

Citations carry an optional ``quote`` field with the snippet of the
chunk the claim relies on; this is informational for v0.1 (not yet
verified against the chunk text by any validator).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A pointer to a single retrieved chunk supporting a claim."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(
        ...,
        min_length=1,
        description="Stable document identifier (e.g. 'doc_00001').",
    )
    chunk_idx_in_doc: int = Field(
        ...,
        ge=0,
        description="Zero-based chunk index inside the document.",
    )
    quote: Optional[str] = Field(
        default=None,
        description=(
            "Optional verbatim snippet from the cited chunk that the "
            "claim relies on. Informational for v0.1; not yet verified "
            "against the chunk text by any validator."
        ),
    )


class Claim(BaseModel):
    """A single factual statement, backed by at least one citation."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description="The claim as a self-contained natural-language statement.",
    )
    citations: List[Citation] = Field(
        ...,
        min_length=1,
        description="One or more citations supporting this claim.",
    )


class Answer(BaseModel):
    """Structured LLM answer: a free-form summary plus cited claims."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "Free-form natural-language answer to the user's question, "
            "intended for direct display. Must be consistent with the "
            "structured claims below."
        ),
    )
    claims: List[Claim] = Field(
        ...,
        min_length=1,
        description="The structured, individually citable claims.",
    )
