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

from typing import Any, Iterable, List

from .llm_client import LLMBackend, LLMResponse


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
