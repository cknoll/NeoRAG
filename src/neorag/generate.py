"""Unstructured generation step for the NeoRAG pipeline.

This module wires retrieval to an LLM backend so that ``neorag query``
returns an actual LLM-authored answer grounded in retrieved chunks
(see __gitignore__improvement-plan2.md, step 1).

The prompt rendering here is deliberately minimal and unstructured:
each retrieved chunk is prefixed with a compact identifier of the form
``[doc_id#chunk_idx_in_doc]`` so that the upcoming structured-output
work (step 2) can ask the LLM to cite by exactly those identifiers
without redoing the prompt scaffolding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from pydantic import ValidationError

from .llm_client import LLMBackend, LLMResponse
from .validate.schema import Answer
from .validate.utils import _extract_json_object


# TODO: check if consistent language (system-prompt and data) improves performance
SYSTEM_PROMPT = (
    "You are a careful retrieval-augmented assistant. "
    "Answer the user's question using ONLY the information contained "
    "in the provided context chunks. Each chunk is prefixed with a "
    "bracketed identifier of the form [doc_id#chunk_idx]. "
    "If the context does not contain enough information to answer, "
    "say so explicitly instead of guessing. "
    "Answer in the same language as the user's question."
)


def _chunk_id(node: Any) -> str:
    """Return a compact ``[doc_id#chunk_idx_in_doc]`` identifier for ``node``.

    Falls back to ``[source#chunk_idx]`` (and ultimately ``[unknown]``) when
    full provenance metadata is not available, so this function is safe to
    call on legacy corpora that have not been built via
    :mod:`neorag.build_corpus`.
    """
    md = getattr(node, "metadata", None) or {}
    doc_id = md.get("doc_id") or md.get("source") or "unknown"
    # Strip a trailing ``.md`` so ``doc_00001.md`` renders as ``doc_00001``.
    if isinstance(doc_id, str) and doc_id.endswith(".md"):
        doc_id = doc_id[: -len(".md")]
    chunk_idx = md.get("chunk_idx_in_doc")
    if chunk_idx is None:
        chunk_idx = md.get("chunk_idx")
    if chunk_idx is None:
        return f"[{doc_id}]"
    return f"[{doc_id}#{chunk_idx}]"


def _node_text(node: Any) -> str:
    """Extract the text payload from a llama_index node-or-NodeWithScore."""
    text = getattr(node, "text", None)
    if text is not None:
        return text
    inner = getattr(node, "node", None)
    if inner is not None and hasattr(inner, "text"):
        return inner.text
    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        return get_content()
    return ""


def render_context(retrieved_nodes: Iterable[Any]) -> str:
    """Render retrieved nodes as a single context block for the prompt."""
    parts: List[str] = []
    for node in retrieved_nodes:
        cid = _chunk_id(node)
        text = _node_text(node).strip()
        parts.append(f"{cid}\n{text}")
    return "\n\n".join(parts)


def build_prompt(query: str, retrieved_nodes: Iterable[Any]) -> List[dict]:
    """Build an OpenAI-style messages list for ``query`` + retrieved context.

    The returned list has two messages: a fixed system prompt and a user
    message containing the rendered context followed by the question.
    """
    nodes = list(retrieved_nodes)
    context = render_context(nodes)
    if context:
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    else:
        user_content = f"Question: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_answer(
    query: str,
    retrieved_nodes: Iterable[Any],
    llm: LLMBackend,
) -> str:
    """Run the LLM on ``query`` + retrieved context and return the text answer.

    For step 1 the answer is free-form text. Structured output (Pydantic
    schema with citations) is added in step 2.
    """
    messages = build_prompt(query, retrieved_nodes)
    response: LLMResponse = llm.chat(messages)
    return response.content


# ---------------------------------------------------------------------------
# Structured generation (improvement-plan2 step 2)
# ---------------------------------------------------------------------------

STRUCTURED_SYSTEM_PROMPT = (
    "You are a careful retrieval-augmented assistant. "
    "Answer the user's question using ONLY the information contained "
    "in the provided context chunks. Each chunk is prefixed with a "
    "bracketed identifier of the form [doc_id#chunk_idx]. "
    "If the context does not contain enough information to answer, "
    "say so explicitly in the 'summary' field and emit a single claim "
    "stating that the context is insufficient (still cite the most "
    "relevant chunk).\n\n"
    "You MUST reply with a single JSON object and nothing else "
    "(no markdown, no code fences, no commentary). The JSON object "
    "MUST conform exactly to this schema:\n"
    "{\n"
    '  "summary": "<free-form natural-language answer in the user\'s language>",\n'
    '  "claims": [\n'
    "    {\n"
    '      "text": "<a single self-contained factual statement>",\n'
    '      "citations": [\n'
    "        {\n"
    '          "doc_id": "<doc id from a [doc_id#chunk_idx] tag>",\n'
    '          "chunk_idx_in_doc": <integer chunk index from the same tag>,\n'
    '          "quote": "<optional verbatim snippet from the cited chunk>"\n'
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Every claim MUST have at least one citation, and each citation's "
    "(doc_id, chunk_idx_in_doc) MUST correspond to one of the "
    "[doc_id#chunk_idx] tags in the provided context. "
    "Answer in the same language as the user's question."
)


@dataclass
class StructuredGenerationResult:
    """Outcome of a structured generation attempt.

    Attributes
    ----------
    raw_text:
        The exact text returned by the LLM backend, unmodified.
    parsed:
        The parsed :class:`Answer` if both JSON decoding and Pydantic
        validation succeeded, else ``None``.
    parse_error:
        Human-readable description of the failure mode when ``parsed``
        is ``None``. ``None`` on success. Surfaced (not swallowed) so
        the structural validator (step 3) can convert it directly into
        a ``Violation``.
    """

    raw_text: str
    parsed: Optional[Answer]
    parse_error: Optional[str]


def build_structured_prompt(query: str, retrieved_nodes: Iterable[Any]) -> List[dict]:
    """Build messages instructing the LLM to emit :class:`Answer` JSON."""
    nodes = list(retrieved_nodes)
    context = render_context(nodes)
    if context:
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Reply with a single JSON object as specified."
        )
    else:
        user_content = (
            f"Question: {query}\n\n"
            "Reply with a single JSON object as specified."
        )

    return [
        {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_structured_response(raw: str) -> StructuredGenerationResult:
    """Parse a raw LLM string into a :class:`StructuredGenerationResult`.

    Factored out of :func:`generate_structured_answer` so the refinement
    loop can reuse it without rebuilding the full prompt.
    """
    candidate = _extract_json_object(raw)
    if candidate is None:
        return StructuredGenerationResult(
            raw_text=raw,
            parsed=None,
            parse_error="No JSON object found in LLM response.",
        )

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        return StructuredGenerationResult(
            raw_text=raw,
            parsed=None,
            parse_error=f"JSON decoding failed: {e}",
        )

    try:
        answer = Answer.model_validate(data)
    except ValidationError as e:
        return StructuredGenerationResult(
            raw_text=raw,
            parsed=None,
            parse_error=f"Schema validation failed: {e}",
        )

    return StructuredGenerationResult(
        raw_text=raw,
        parsed=answer,
        parse_error=None,
    )


def generate_structured_answer(
    query: str,
    retrieved_nodes: Iterable[Any],
    llm: LLMBackend,
) -> StructuredGenerationResult:
    """Run the LLM and try to parse its reply as an :class:`Answer`.

    Parsing failures (no JSON object found, malformed JSON, schema
    violations) are surfaced via ``StructuredGenerationResult.parse_error``
    rather than swallowed -- the structural validator added in step 3
    will turn them into ``Violation`` objects.
    """
    messages = build_structured_prompt(query, retrieved_nodes)
    response: LLMResponse = llm.chat(messages)
    return _parse_structured_response(response.content or "")
