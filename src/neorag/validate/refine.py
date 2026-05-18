"""Self-refinement loop for structured NeoRAG answers (improvement-plan2 step 4).

The :func:`refine` function iteratively re-prompts the LLM with its
previous (flawed) answer plus a list of constraint violations until the
answer is violation-free or ``max_iter`` is reached.

``feedback_granularity`` is the experimental factor from AP4.3:
  - ``"coarse"``        — one summary line (total count + violation kinds)
  - ``"per_violation"`` — each violation listed individually with location
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Literal, Optional, Tuple

from ..generate import STRUCTURED_SYSTEM_PROMPT, _parse_structured_response, render_context
from ..llm_client import LLMBackend
from .groundedness import validate_groundedness
from .schema import Answer
from .shacl import validate_shacl
from .violation import Violation


@dataclass
class RefinementIteration:
    """Record of one refinement attempt.

    Attributes
    ----------
    iteration:
        1-based counter.
    violations_in:
        Violations that triggered this refinement call.
    feedback:
        Feedback text sent to the LLM for this iteration.
    raw_response:
        Verbatim LLM output for this iteration.
    answer:
        Parsed :class:`Answer`, or ``None`` if the LLM response could
        not be parsed (structural failure).
    violations_out:
        Violations in this iteration's answer. Empty when ``answer``
        is ``None`` (structural failure) or when the answer is clean.
    """

    iteration: int
    violations_in: List[Violation]
    feedback: str
    raw_response: str
    answer: Optional[Answer]
    violations_out: List[Violation] = field(default_factory=list)


def _build_feedback(violations: List[Violation], granularity: str) -> str:
    """Format ``violations`` as a feedback string for the LLM.

    ``"coarse"`` produces a single summary line; ``"per_violation"``
    lists each violation with its kind, optional location, and message.
    """
    if granularity == "coarse":
        kinds = sorted({v.kind for v in violations})
        return (
            f"Found {len(violations)} violation(s) "
            f"(kinds: {', '.join(kinds)}). "
            "Please revise your answer to fix all issues."
        )
    # per_violation (default)
    lines = [f"Found {len(violations)} violation(s):"]
    for v in violations:
        loc = f" {v.location}:" if v.location else ""
        lines.append(f"  - [{v.kind}]{loc} {v.message}")
    return "\n".join(lines)


def _build_refinement_prompt(
    query: str,
    retrieved_nodes: List[Any],
    current_answer: Answer,
    feedback: str,
) -> List[dict]:
    """Build an OpenAI-style messages list for one refinement step.

    Includes the retrieved context (so the LLM can re-cite correctly),
    the previous answer as JSON, and the feedback about what was wrong.
    """
    context = render_context(retrieved_nodes)
    prev_json = current_answer.model_dump_json(indent=2)

    if context:
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Your previous answer:\n{prev_json}\n\n"
            f"Issues found:\n{feedback}\n\n"
            "Provide a corrected answer as a single JSON object conforming to the schema."
        )
    else:
        user_content = (
            f"Question: {query}\n\n"
            f"Your previous answer:\n{prev_json}\n\n"
            f"Issues found:\n{feedback}\n\n"
            "Provide a corrected answer as a single JSON object conforming to the schema."
        )

    return [
        {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def refine(
    query: str,
    retrieved_nodes: Iterable[Any],
    answer: Answer,
    violations: List[Violation],
    llm: LLMBackend,
    max_iter: int = 3,
    feedback_granularity: Literal["coarse", "per_violation"] = "per_violation",
) -> Tuple[Answer, List[RefinementIteration]]:
    """Iteratively refine ``answer`` until all violations are resolved.

    Parameters
    ----------
    query:
        Original user query.
    retrieved_nodes:
        Retrieved context nodes (materialised once and reused across
        all iterations).
    answer:
        Initial structured answer to start refining from.
    violations:
        Constraint violations in the initial answer.
    llm:
        LLM backend used for each refinement call.
    max_iter:
        Maximum number of refinement iterations. The loop exits early
        if a violation-free answer is produced before this limit.
    feedback_granularity:
        How violations are described in the refinement prompt.
        ``"coarse"`` — one summary line; ``"per_violation"`` — each
        violation listed individually. This is the experimental factor
        from AP4.3.

    Returns
    -------
    (final_answer, history) :
        ``final_answer`` is the last successfully parsed :class:`Answer`
        (falling back to the original ``answer`` if every iteration
        failed to parse). ``history`` is the list of
        :class:`RefinementIteration` records, one per iteration
        actually executed. Returns ``(answer, [])`` immediately —
        without calling the LLM — when ``violations`` is empty.
    """
    nodes = list(retrieved_nodes)

    if not violations:
        return answer, []

    history: List[RefinementIteration] = []
    current_answer = answer
    current_violations: List[Violation] = list(violations)

    for i in range(1, max_iter + 1):
        feedback = _build_feedback(current_violations, feedback_granularity)
        messages = _build_refinement_prompt(query, nodes, current_answer, feedback)

        response = llm.chat(messages)
        raw = response.content or ""
        result = _parse_structured_response(raw)

        if result.parsed is None:
            history.append(
                RefinementIteration(
                    iteration=i,
                    violations_in=list(current_violations),
                    feedback=feedback,
                    raw_response=raw,
                    answer=None,
                    violations_out=[],
                )
            )
            # Keep current_answer unchanged; carry the structural error forward.
            current_violations = [
                Violation(
                    kind="structural",
                    message=result.parse_error or "Unknown parsing error.",
                )
            ]
        else:
            new_violations = validate_groundedness(result.parsed, nodes) + validate_shacl(
                result.parsed, nodes
            )
            history.append(
                RefinementIteration(
                    iteration=i,
                    violations_in=list(current_violations),
                    feedback=feedback,
                    raw_response=raw,
                    answer=result.parsed,
                    violations_out=new_violations,
                )
            )
            current_answer = result.parsed
            current_violations = new_violations
            if not new_violations:
                break

    return current_answer, history
