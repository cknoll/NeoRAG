"""Tests for evaluate.py (improvement-plan2 step 6)."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

from neorag.evaluate import QueryRecord, TokenCountingBackend, _serialize, evaluate
from neorag.llm_client import LLMResponse, StubBackend


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeNode:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.9


class MockPipeline:
    """Returns a fixed list of FakeNodes for every query."""

    def __init__(self, nodes: List[FakeNode]):
        self._nodes = nodes
        self.queries_seen: List[str] = []

    def retrieve(self, query: str) -> List[FakeNode]:
        self.queries_seen.append(query)
        return self._nodes


class SequenceStubBackend:
    """Returns responses from ``responses`` in order; repeats the last."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.call_count = 0

    def chat(self, messages, tools=None):
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return LLMResponse(content=self.responses[idx], usage={"prompt_tokens": 10, "completion_tokens": 5})


def _grounded_json(doc_id: str = "doc_00001", chunk_idx: int = 0) -> str:
    return json.dumps({
        "summary": "Grounded answer.",
        "claims": [
            {
                "text": "A claim.",
                "citations": [{"doc_id": doc_id, "chunk_idx_in_doc": chunk_idx}],
            }
        ],
    })


def _ungrounded_json() -> str:
    return json.dumps({
        "summary": "Hallucinated.",
        "claims": [
            {
                "text": "Bogus.",
                "citations": [{"doc_id": "doc_99999", "chunk_idx_in_doc": 0}],
            }
        ],
    })


def _write_test_set(tmp_path: Path, items: list) -> Path:
    p = tmp_path / "test_set.json"
    p.write_text(json.dumps(items))
    return p


def _default_nodes():
    return [
        FakeNode("chunk 0", {"doc_id": "doc_00001", "source": "doc_00001.md", "chunk_idx_in_doc": 0}),
        FakeNode("chunk 1", {"doc_id": "doc_00001", "source": "doc_00001.md", "chunk_idx_in_doc": 1}),
    ]


# ---------------------------------------------------------------------------
# _serialize
# ---------------------------------------------------------------------------


class TestSerialize(unittest.TestCase):
    def test_scalars_pass_through(self):
        self.assertEqual(_serialize(42), 42)
        self.assertEqual(_serialize("x"), "x")
        self.assertIsNone(_serialize(None))
        self.assertEqual(_serialize(True), True)

    def test_list_recursed(self):
        self.assertEqual(_serialize([1, "a", None]), [1, "a", None])

    def test_dict_recursed(self):
        self.assertEqual(_serialize({"a": 1}), {"a": 1})

    def test_dataclass_serialized(self):
        from neorag.validate.violation import Violation
        v = Violation(kind="structural", message="oops", location="c[0]")
        result = _serialize(v)
        self.assertEqual(result, {"kind": "structural", "message": "oops", "location": "c[0]"})

    def test_pydantic_model_serialized(self):
        from neorag.validate.schema import Answer, Citation, Claim
        ans = Answer(
            summary="s",
            claims=[Claim(text="t", citations=[Citation(doc_id="d", chunk_idx_in_doc=0)])],
        )
        result = _serialize(ans)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["summary"], "s")


# ---------------------------------------------------------------------------
# TokenCountingBackend
# ---------------------------------------------------------------------------


class TestTokenCountingBackend(unittest.TestCase):
    def test_counts_accumulate(self):
        inner = StubBackend(canned_response="x")
        # Patch usage into the response
        import unittest.mock as mock
        with mock.patch.object(inner, "chat", return_value=LLMResponse(
            content="x", usage={"prompt_tokens": 10, "completion_tokens": 5}
        )):
            backend = TokenCountingBackend(inner)
            backend.chat([])
            backend.chat([])
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(backend.prompt_tokens, 20)
        self.assertEqual(backend.completion_tokens, 10)

    def test_reset_clears_counts(self):
        inner = StubBackend(canned_response="x")
        backend = TokenCountingBackend(inner)
        backend.call_count = 5
        backend.prompt_tokens = 100
        backend.reset()
        self.assertEqual(backend.call_count, 0)
        self.assertEqual(backend.prompt_tokens, 0)

    def test_missing_usage_does_not_crash(self):
        inner = StubBackend(canned_response="x")
        backend = TokenCountingBackend(inner)
        backend.chat([])  # StubBackend returns usage=None
        self.assertEqual(backend.prompt_tokens, 0)


# ---------------------------------------------------------------------------
# evaluate() — retrieval metrics
# ---------------------------------------------------------------------------


class TestEvaluateRetrievalMetrics(unittest.TestCase):
    def _nodes_with_source(self, source: str):
        return [FakeNode("text", {"source": source, "doc_id": source, "chunk_idx_in_doc": 0})]

    def test_mrr_rank_1_hit(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [{"query": "q1", "expected_sources": ["doc_a.md"]}]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(self._nodes_with_source("doc_a.md"))
        backend = StubBackend(canned_response=_grounded_json())

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline, no_refine=True,
        )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        self.assertAlmostEqual(metrics["mrr_at_10"], 1.0)
        self.assertAlmostEqual(metrics["recall_at_5"], 1.0)

    def test_mrr_miss(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [{"query": "q1", "expected_sources": ["other.md"]}]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(self._nodes_with_source("doc_a.md"))
        backend = StubBackend(canned_response=_grounded_json())

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline, no_refine=True,
        )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        self.assertAlmostEqual(metrics["mrr_at_10"], 0.0)
        self.assertAlmostEqual(metrics["recall_at_5"], 0.0)

    def test_mrr_averages_across_queries(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        # q1: hit at rank 1 (MRR=1.0), q2: miss (MRR=0.0) → mean = 0.5
        test_set = [
            {"query": "q1", "expected_sources": ["doc_a.md"]},
            {"query": "q2", "expected_sources": ["missing.md"]},
        ]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(self._nodes_with_source("doc_a.md"))
        backend = StubBackend(canned_response=_grounded_json())

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline, no_refine=True,
        )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        self.assertAlmostEqual(metrics["mrr_at_10"], 0.5)


# ---------------------------------------------------------------------------
# evaluate() — generation + violation metrics
# ---------------------------------------------------------------------------


class TestEvaluateViolationMetrics(unittest.TestCase):
    def test_grounded_answer_zero_violation_rate(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [{"query": "q1", "expected_sources": ["doc_00001.md"]}]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(_default_nodes())
        backend = StubBackend(canned_response=_grounded_json())

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline, no_refine=True,
        )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        self.assertAlmostEqual(metrics["violation_rate_before"], 0.0)
        self.assertAlmostEqual(metrics["violation_rate_after"], 0.0)
        self.assertAlmostEqual(metrics["mean_iterations"], 0.0)

    def test_ungrounded_then_grounded_reduces_violation_rate(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [{"query": "q1", "expected_sources": ["doc_00001.md"]}]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(_default_nodes())
        # First call: ungrounded (triggers refinement); second call: grounded.
        backend = SequenceStubBackend([_ungrounded_json(), _grounded_json()])

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline,
            max_iter=3, no_refine=False,
        )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        self.assertAlmostEqual(metrics["violation_rate_before"], 1.0)
        self.assertAlmostEqual(metrics["violation_rate_after"], 0.0)
        self.assertAlmostEqual(metrics["mean_iterations"], 1.0)

    def test_no_refine_flag_leaves_violations_after_unchanged(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [{"query": "q1", "expected_sources": ["doc_00001.md"]}]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(_default_nodes())
        backend = StubBackend(canned_response=_ungrounded_json())

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline,
            no_refine=True,
        )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        self.assertAlmostEqual(metrics["violation_rate_before"], 1.0)
        # no refinement → violations_after == violations_before
        self.assertAlmostEqual(metrics["violation_rate_after"], 1.0)
        self.assertAlmostEqual(metrics["mean_iterations"], 0.0)


# ---------------------------------------------------------------------------
# evaluate() — runs/ directory structure
# ---------------------------------------------------------------------------


class TestEvaluateRunsStructure(unittest.TestCase):
    def test_creates_config_and_metrics_json(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [{"query": "q1", "expected_sources": ["x"]}]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(_default_nodes())
        backend = StubBackend(canned_response=_grounded_json())

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline, no_refine=True,
        )
        self.assertTrue((run_dir / "config.json").is_file())
        self.assertTrue((run_dir / "metrics.json").is_file())
        self.assertTrue((run_dir / "queries").is_dir())

    def test_config_json_contains_expected_fields(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [{"query": "q1", "expected_sources": ["x"]}]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(_default_nodes())
        backend = StubBackend(canned_response=_grounded_json())

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline,
            max_iter=5, feedback_granularity="coarse", no_refine=True,
        )
        cfg = json.loads((run_dir / "config.json").read_text())
        self.assertEqual(cfg["max_iter"], 5)
        self.assertEqual(cfg["feedback_granularity"], "coarse")
        self.assertTrue(cfg["no_refine"])
        self.assertIn("timestamp", cfg)
        self.assertIn("test_path", cfg)

    def test_one_query_file_per_test_item(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [
            {"query": "q1", "expected_sources": ["x"]},
            {"query": "q2", "expected_sources": ["y"]},
        ]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(_default_nodes())
        backend = StubBackend(canned_response=_grounded_json())

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline, no_refine=True,
        )
        query_files = sorted((run_dir / "queries").iterdir())
        self.assertEqual(len(query_files), 2)
        self.assertEqual(query_files[0].name, "query_0000.json")
        self.assertEqual(query_files[1].name, "query_0001.json")

    def test_query_json_contains_expected_fields(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [{"query": "What is X?", "expected_sources": ["doc_00001.md"]}]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline(_default_nodes())
        backend = StubBackend(canned_response=_grounded_json())

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline, no_refine=True,
        )
        qdata = json.loads((run_dir / "queries" / "query_0000.json").read_text())
        self.assertEqual(qdata["query"], "What is X?")
        self.assertIn("mrr_at_10", qdata)
        self.assertIn("recall_at_5", qdata)
        self.assertIn("latency_s", qdata)
        self.assertIn("initial_raw_response", qdata)
        self.assertIn("violations_before", qdata)
        self.assertIn("violations_after", qdata)
        self.assertIn("n_iterations", qdata)
        self.assertIn("refinement_history", qdata)

    def test_run_dir_name_is_timestamp_format(self, tmp_path=None):
        tmp_path = tmp_path or Path(pytest.importorskip("tempfile").mkdtemp())
        test_set = [{"query": "q", "expected_sources": []}]
        test_path = _write_test_set(Path(tmp_path), test_set)
        pipeline = MockPipeline([])
        backend = StubBackend(canned_response="garbage")

        run_dir = evaluate(
            test_path, runs_dir=Path(tmp_path) / "runs",
            _llm_backend=backend, _pipeline=pipeline, no_refine=True,
        )
        import re
        self.assertRegex(run_dir.name, r"^\d{8}_\d{6}$")
