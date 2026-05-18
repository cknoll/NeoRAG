"""Groundedness validator.

Every :class:`Citation` in the parsed answer must reference a
``(doc_id, chunk_idx_in_doc)`` pair that was actually present in the
retrieved context for this query. Anything else is treated as a
hallucinated / uncited reference.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Set, Tuple

from .schema import Answer
from .violation import Violation


def _retrieved_chunk_ids(
    retrieved_nodes: Iterable[Any],
) -> Set[Tuple[str, int]]:
    """Extract the set of ``(doc_id, chunk_idx_in_doc)`` pairs from nodes.

    Falls back to ``source`` (with a trailing ``.md`` stripped) and
    ``chunk_idx`` for legacy nodes that do not carry full provenance.
    Nodes that cannot yield both a doc id and a chunk index are skipped.
    """
    ids: Set[Tuple[str, int]] = set()
    for node in retrieved_nodes:
        md = getattr(node, "metadata", None) or {}
        doc_id = md.get("doc_id") or md.get("source")
        if isinstance(doc_id, str) and doc_id.endswith(".md"):
            doc_id = doc_id[: -len(".md")]
        chunk_idx = md.get("chunk_idx_in_doc")
        if chunk_idx is None:
            chunk_idx = md.get("chunk_idx")
        if doc_id is None or chunk_idx is None:
            continue
        try:
            ids.add((str(doc_id), int(chunk_idx)))
        except (TypeError, ValueError):
            continue
    return ids


def validate_groundedness(
    answer: Answer,
    retrieved_nodes: Iterable[Any],
) -> List[Violation]:
    """Check that every citation in ``answer`` is grounded in retrieval.

    Returns a list of ``Violation(kind="groundedness", ...)``. Empty
    list means every citation references a chunk that was actually
    retrieved.
    """
    valid_ids = _retrieved_chunk_ids(retrieved_nodes)
    violations: List[Violation] = []

    for ci, claim in enumerate(answer.claims):
        for cj, cit in enumerate(claim.citations):
            key = (cit.doc_id, cit.chunk_idx_in_doc)
            if key not in valid_ids:
                violations.append(
                    Violation(
                        kind="groundedness",
                        message=(
                            f"Citation ({cit.doc_id}, "
                            f"chunk_idx_in_doc={cit.chunk_idx_in_doc}) "
                            "does not match any retrieved chunk."
                        ),
                        location=f"claims[{ci}].citations[{cj}]",
                    )
                )
    return violations
