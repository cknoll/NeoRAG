"""Tests for the self-refinement loop (improvement-plan2 step 4)."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neorag.llm_client import LLMResponse
from neorag.validate.refine import (
    RefinementIteration,
    _build_feedback,
    _build_refinement_prompt,
    refine,
)
from neorag.validate.schema import Answer, Citation, Claim
from neorag.validate.violation import Violation


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeNode:
    text: str
    metadata: dict


def _nodes():
    return [
        FakeNode("Chunk zero.", {"doc_id": "doc_00001", "chunk_idx_in_doc": 0}),
        FakeNode("Chunk one.", {"doc_id": "doc_00001", "chunk_idx_in_doc": 1}),
    ]


def _good_answer() -> Answer:
    return Answer.model_validate(
        {
            "summary": "Grounded answer.",
            "claims": [
                {
                    "text": "Claim citing chunk zero.",
                    "citations": [{"doc_id": "doc_00001", "chunk_idx_in_doc": 0}],
                }
            ],
        }
    )


def _bad_answer() -> Answer:
    """Answer with a hallucinated citation (not in _nodes())."""
    return Answer.model_validate(
        {
            "summary": "Hallucinated.",
            "claims": [
                {
                    "text": "Bogus claim.",
                    "citations": [{"doc_id": "doc_99999", "chunk_idx_in_doc": 0}],
                }
            ],
        }
    )


def _good_json() -> str:
    return json.dumps(
        {
            "summary": "Grounded answer.",
            "claims": [
                {
                    "text": "Claim citing chunk zero.",
                    "citations": [{"doc_id": "doc_00001", "chunk_idx_in_doc": 0}],
                }
            ],
        }
    )


def _bad_json() -> str:
    return json.dumps(
        {
            "summary": "Still hallucinated.",
            "claims": [
                {
                    "text": "Bogus claim.",
                    "citations": [{"doc_id": "doc_99999", "chunk_idx_in_doc": 0}],
                }
            ],
        }
    )


def _groundedness_violation() -> Violation:
    return Violation(
        kind="groundedness",
        message="Citation (doc_99999, chunk_idx_in_doc=0) does not match any retrieved chunk.",
        location="claims[0].citations[0]",
    )


class SequenceStubBackend:
    """Returns responses from ``responses`` in order; repeats the last."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.call_count = 0
        self.messages_history: List[List[Dict[str, Any]]] = []

    def chat(self, messages, tools=None):
        self.messages_history.append(messages)
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return LLMResponse(content=self.responses[idx])


# ---------------------------------------------------------------------------
# _build_feedback
# ---------------------------------------------------------------------------


class TestBuildFeedback(unittest.TestCase):
    def _violations(self):
        return [
            Violation(kind="groundedness", message="bad cite", location="claims[0].citations[0]"),
            Violation(kind="shacl", message="shacl failed"),
        ]

    def test_coarse_single_line_with_kinds(self):
        fb = _build_feedback(self._violations(), "coarse")
        self.assertIn("2 violation(s)", fb)
        self.assertIn("groundedness", fb)
        self.assertIn("shacl", fb)
        # coarse should be one line only
        self.assertEqual(fb.count("\n"), 0)

    def test_per_violation_lists_each(self):
        fb = _build_feedback(self._violations(), "per_violation")
        self.assertIn("2 violation(s)", fb)
        self.assertIn("[groundedness]", fb)
        self.assertIn("[shacl]", fb)
        self.assertIn("claims[0].citations[0]", fb)
        # header line + 2 violation lines = 3 lines total
        self.assertEqual(fb.count("\n"), 2)

    def test_per_violation_without_location(self):
        fb = _build_feedback(
            [Violation(kind="structural", message="no JSON")], "per_violation"
        )
        self.assertIn("[structural]", fb)
        self.assertIn("no JSON", fb)

    def test_kinds_in_coarse_are_sorted(self):
        violations = [
            Violation(kind="shacl", message="s"),
            Violation(kind="groundedness", message="g"),
        ]
        fb = _build_feedback(violations, "coarse")
        idx_g = fb.index("groundedness")
        idx_s = fb.index("shacl")
        self.assertLess(idx_g, idx_s)


# ---------------------------------------------------------------------------
# _build_refinement_prompt
# ---------------------------------------------------------------------------


class TestBuildRefinementPrompt(unittest.TestCase):
    def test_two_messages_system_then_user(self):
        msgs = _build_refinement_prompt("What?", _nodes(), _good_answer(), "fix this")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")

    def test_user_message_contains_context_and_query(self):
        msgs = _build_refinement_prompt("What is X?", _nodes(), _good_answer(), "fix this")
        user = msgs[1]["content"]
        self.assertIn("[doc_00001#0]", user)
        self.assertIn("What is X?", user)

    def test_user_message_contains_previous_answer_json(self):
        msgs = _build_refinement_prompt("Q?", _nodes(), _good_answer(), "fix this")
        user = msgs[1]["content"]
        # The previous answer's summary should appear as a JSON field value
        self.assertIn("Grounded answer.", user)
        self.assertIn("doc_00001", user)

    def test_user_message_contains_feedback(self):
        msgs = _build_refinement_prompt("Q?", _nodes(), _good_answer(), "MY SPECIAL FEEDBACK")
        self.assertIn("MY SPECIAL FEEDBACK", msgs[1]["content"])

    def test_empty_context_still_includes_query_and_answer(self):
        msgs = _build_refinement_prompt("Q?", [], _good_answer(), "fix")
        user = msgs[1]["content"]
        self.assertIn("Q?", user)
        self.assertIn("Grounded answer.", user)
        self.assertNotIn("Context:", user)


# ---------------------------------------------------------------------------
# refine()
# ---------------------------------------------------------------------------


class TestRefineNoViolations(unittest.TestCase):
    def test_returns_answer_immediately_without_calling_llm(self):
        backend = SequenceStubBackend([_good_json()])
        final, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_good_answer(),
            violations=[],
            llm=backend,
        )
        self.assertEqual(backend.call_count, 0)
        self.assertEqual(history, [])
        self.assertEqual(final.summary, "Grounded answer.")


class TestRefineSuccessOnFirstRetry(unittest.TestCase):
    def test_one_iteration_history_and_clean_final_answer(self):
        backend = SequenceStubBackend([_good_json()])
        final, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            max_iter=3,
        )
        self.assertEqual(backend.call_count, 1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].iteration, 1)
        self.assertEqual(history[0].violations_in, [_groundedness_violation()])
        self.assertEqual(history[0].violations_out, [])
        self.assertIsNotNone(history[0].answer)
        self.assertEqual(final.summary, "Grounded answer.")

    def test_history_records_raw_response(self):
        raw = _good_json()
        backend = SequenceStubBackend([raw])
        _, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
        )
        self.assertEqual(history[0].raw_response, raw)


class TestRefineMaxIter(unittest.TestCase):
    def test_structural_failures_exhaust_max_iter(self):
        backend = SequenceStubBackend(["this is not json"])
        final, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            max_iter=2,
        )
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(len(history), 2)
        # All iterations failed to parse
        for rec in history:
            self.assertIsNone(rec.answer)
            self.assertEqual(rec.violations_out, [])
        # Falls back to the initial answer
        self.assertEqual(final.summary, "Hallucinated.")

    def test_persistent_violations_exhaust_max_iter(self):
        # LLM keeps returning an answer with violations
        backend = SequenceStubBackend([_bad_json()])
        final, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            max_iter=2,
        )
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(len(history), 2)
        # Each iteration parsed but still has violations
        for rec in history:
            self.assertIsNotNone(rec.answer)
            self.assertTrue(len(rec.violations_out) >= 1)

    def test_iteration_numbers_are_one_based(self):
        backend = SequenceStubBackend(["garbage"])
        _, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            max_iter=3,
        )
        self.assertEqual([r.iteration for r in history], [1, 2, 3])

    def test_max_iter_zero_returns_original_answer(self):
        backend = SequenceStubBackend([_good_json()])
        final, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            max_iter=0,
        )
        self.assertEqual(backend.call_count, 0)
        self.assertEqual(history, [])
        self.assertEqual(final.summary, "Hallucinated.")


class TestRefineFeedbackGranularity(unittest.TestCase):
    def test_coarse_feedback_sent_to_llm(self):
        backend = SequenceStubBackend([_good_json()])
        refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            feedback_granularity="coarse",
        )
        user_msg = backend.messages_history[0][1]["content"]
        self.assertIn("violation(s)", user_msg)
        self.assertIn("groundedness", user_msg)
        # "coarse" feedback should not list individual violation details
        self.assertNotIn("claims[0]", user_msg)

    def test_per_violation_feedback_lists_location(self):
        backend = SequenceStubBackend([_good_json()])
        refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            feedback_granularity="per_violation",
        )
        user_msg = backend.messages_history[0][1]["content"]
        self.assertIn("claims[0].citations[0]", user_msg)

    def test_feedback_granularity_recorded_in_history(self):
        backend = SequenceStubBackend([_good_json()])
        _, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            feedback_granularity="coarse",
        )
        # The recorded feedback should be the coarse variant
        self.assertNotIn("claims[0]", history[0].feedback)

    def test_default_granularity_is_per_violation(self):
        backend = SequenceStubBackend([_good_json()])
        _, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
        )
        self.assertIn("claims[0].citations[0]", history[0].feedback)


class TestRefineStructuralRecovery(unittest.TestCase):
    def test_structural_then_success(self):
        """First retry fails to parse; second retry succeeds."""
        backend = SequenceStubBackend(["not json", _good_json()])
        final, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            max_iter=3,
        )
        self.assertEqual(len(history), 2)
        self.assertIsNone(history[0].answer)
        self.assertIsNotNone(history[1].answer)
        self.assertEqual(history[1].violations_out, [])
        self.assertEqual(final.summary, "Grounded answer.")

    def test_structural_failure_carries_error_to_next_violations_in(self):
        """After a structural failure, the next iteration sees a structural violation."""
        backend = SequenceStubBackend(["not json", _good_json()])
        _, history = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            max_iter=3,
        )
        self.assertEqual(history[1].violations_in[0].kind, "structural")

    def test_current_answer_unchanged_after_structural_failure(self):
        """If parsing fails, final answer falls back to the last valid one."""
        # Two structural failures, so final answer = initial _bad_answer
        backend = SequenceStubBackend(["bad", "bad"])
        final, _ = refine(
            query="Q?",
            retrieved_nodes=_nodes(),
            answer=_bad_answer(),
            violations=[_groundedness_violation()],
            llm=backend,
            max_iter=2,
        )
        self.assertEqual(final.summary, "Hallucinated.")
