"""Evaluation module for NeoRAG.

Measures retrieval quality (MRR@10, Recall@5) and — when a LLM backend is
available — generation quality (constraint-violation rates before/after
refinement, refinement iterations, wall-clock latency).

Each invocation dumps a timestamped directory under ``runs/`` containing:
  - ``config.json``      resolved evaluation settings
  - ``metrics.json``     aggregated metrics
  - ``queries/``         one JSON file per test query (raw outputs + violations)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QueryRecord:
    """All per-query data collected during one evaluation run."""

    query: str
    expected_sources: List[str]
    retrieved_sources: List[str]
    mrr_at_10: float
    recall_at_5: bool
    latency_s: float
    # Generation fields — None when the LLM was not called for this query.
    initial_raw_response: Optional[str] = None
    parse_error: Optional[str] = None
    violations_before: List[Any] = field(default_factory=list)  # List[Violation]
    violations_after: List[Any] = field(default_factory=list)   # List[Violation]
    n_iterations: int = 0
    refinement_history: List[Any] = field(default_factory=list)  # List[RefinementIteration]


class TokenCountingBackend:
    """Wraps an LLMBackend and sums token counts from ``LLMResponse.usage``."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.call_count: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def reset(self) -> None:
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, messages: Any, tools: Any = None) -> Any:
        response = self.inner.chat(messages, tools)
        self.call_count += 1
        if response.usage:
            self.prompt_tokens += response.usage.get("prompt_tokens", 0)
            self.completion_tokens += response.usage.get("completion_tokens", 0)
        return response


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _serialize(obj: Any) -> Any:
    """Recursively convert ``obj`` to a JSON-safe value.

    Handles Pydantic models (via ``model_dump``), dataclasses (via field
    iteration), lists, dicts, and scalars. Falls back to ``str`` for
    anything else.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):          # Pydantic BaseModel
        return _serialize(obj.model_dump())
    if hasattr(obj, "__dataclass_fields__"):  # dataclass
        return {k: _serialize(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return str(obj)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    test_path: Path,
    runs_dir: Path = Path("runs"),
    max_iter: int = 3,
    feedback_granularity: str = "per_violation",
    no_refine: bool = False,
    _llm_backend: Any = None,
    _pipeline: Any = None,
) -> Path:
    """Evaluate retrieval + generation quality on a labelled test set.

    Parameters
    ----------
    test_path:
        Path to a JSON file containing a list of objects with at least
        ``"query"`` and ``"expected_sources"`` fields.
    runs_dir:
        Parent directory for timestamped run folders (created on demand).
    max_iter:
        Maximum refinement iterations per query.
    feedback_granularity:
        ``"coarse"`` or ``"per_violation"`` (see :mod:`neorag.validate.refine`).
    no_refine:
        If ``True``, validate but skip the refinement loop.
    _llm_backend:
        Injected LLM backend for testing. If ``None`` and generation is
        enabled, ``LLMClient.from_config()`` is used.
    _pipeline:
        Injected retrieval pipeline for testing. If ``None``,
        ``get_retrieval_pipeline()`` is used.

    Returns
    -------
    Path
        The newly created run directory under ``runs_dir``.
    """
    from .retriever import get_retrieval_pipeline
    from .generate import generate_structured_answer
    from .validate import validate
    from .validate.refine import refine

    test_data: List[dict] = json.loads(Path(test_path).read_text())
    pipeline = _pipeline if _pipeline is not None else get_retrieval_pipeline()

    # LLM backend — only instantiate when generation is needed.
    llm_raw: Any = _llm_backend
    _we_own_llm = False
    if llm_raw is None:
        from .llm_client import LLMClient
        llm_raw = LLMClient.from_config()
        _we_own_llm = True

    llm = TokenCountingBackend(llm_raw)

    records: List[QueryRecord] = []
    mrr_total = 0.0
    recall_at_5_count = 0

    try:
        for idx, item in enumerate(test_data, 1):
            query: str = item["query"]
            expected_sources: List[str] = list(item["expected_sources"])
            print(f"  [{idx}/{len(test_data)}] {query[:70]}")

            t0 = time.monotonic()
            nodes = pipeline.retrieve(query)
            retrieved_sources = [
                node.metadata.get("source") or node.metadata.get("doc_id", "unknown")
                for node in nodes
            ]

            # MRR@10
            mrr_at_10 = 0.0
            for rank, source in enumerate(retrieved_sources[:10], 1):
                if source in expected_sources:
                    mrr_at_10 = 1.0 / rank
                    mrr_total += mrr_at_10
                    break

            # Recall@5
            recall_at_5 = any(s in expected_sources for s in retrieved_sources[:5])
            if recall_at_5:
                recall_at_5_count += 1

            record = QueryRecord(
                query=query,
                expected_sources=expected_sources,
                retrieved_sources=retrieved_sources,
                mrr_at_10=mrr_at_10,
                recall_at_5=recall_at_5,
                latency_s=0.0,
            )

            llm.reset()
            gen_result = generate_structured_answer(query, nodes, llm)
            record.initial_raw_response = gen_result.raw_text

            if gen_result.parsed is None:
                record.parse_error = gen_result.parse_error
            else:
                violations_before = validate(gen_result.raw_text, nodes)
                record.violations_before = list(violations_before)

                final_answer = gen_result.parsed
                history: list = []

                if violations_before and not no_refine:
                    final_answer, history = refine(
                        query=query,
                        retrieved_nodes=nodes,
                        answer=gen_result.parsed,
                        violations=violations_before,
                        llm=llm,
                        max_iter=max_iter,
                        feedback_granularity=feedback_granularity,
                    )

                record.n_iterations = len(history)
                record.refinement_history = history

                # Violations on the final answer: last parsed iteration's
                # violations_out, or the original set if nothing parsed.
                if history:
                    last_valid = next(
                        (h for h in reversed(history) if h.answer is not None), None
                    )
                    record.violations_after = (
                        list(last_valid.violations_out) if last_valid else list(violations_before)
                    )
                else:
                    record.violations_after = list(violations_before)

            record.latency_s = time.monotonic() - t0
            records.append(record)

    finally:
        if _we_own_llm and hasattr(llm_raw, "close"):
            llm_raw.close()

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    n = len(records)
    generated = [r for r in records if r.initial_raw_response is not None]
    parsed = [r for r in generated if r.parse_error is None]

    metrics: dict = {
        "n_queries": n,
        "mrr_at_10": round(mrr_total / n, 6) if n else 0.0,
        "recall_at_5": round(recall_at_5_count / n, 6) if n else 0.0,
    }
    if generated:
        ng = len(generated)
        np_ = len(parsed)
        metrics["parse_failure_rate"] = round((ng - np_) / ng, 6)
    if parsed:
        np_ = len(parsed)
        metrics["violation_rate_before"] = round(
            sum(1 for r in parsed if r.violations_before) / np_, 6
        )
        metrics["violation_rate_after"] = round(
            sum(1 for r in parsed if r.violations_after) / np_, 6
        )
        metrics["mean_iterations"] = round(
            sum(r.n_iterations for r in parsed) / np_, 6
        )
        metrics["mean_latency_s"] = round(
            sum(r.latency_s for r in records) / n, 6
        )

    # ------------------------------------------------------------------
    # Dump runs/ directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_dir) / timestamp
    queries_dir = run_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "test_path": str(test_path),
        "max_iter": max_iter,
        "feedback_granularity": feedback_granularity,
        "no_refine": no_refine,
        "timestamp": timestamp,
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    for i, record in enumerate(records):
        (queries_dir / f"query_{i:04d}.json").write_text(
            json.dumps(_serialize(record), indent=2)
        )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\nMRR@10:   {metrics['mrr_at_10']:.3f}")
    print(f"Recall@5: {metrics['recall_at_5']:.3f}")
    if parsed:
        print(f"Violation rate before refinement: {metrics['violation_rate_before']:.1%}")
        print(f"Violation rate after  refinement: {metrics['violation_rate_after']:.1%}")
        print(f"Mean refinement iterations:       {metrics['mean_iterations']:.2f}")
        print(f"Mean latency (s):                 {metrics['mean_latency_s']:.2f}")
    print(f"\nRun saved to: {run_dir}")

    return run_dir
