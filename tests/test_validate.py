"""Tests for the validator package.

TODO: handcrafted fixtures live inline for now; relocate to
``tests/fixtures/`` once a second consumer needs them.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from neorag.validate import (
    Violation,
    validate,
    validate_groundedness,
    validate_shacl,
    validate_structural,
)
from neorag.validate.schema import Answer, Citation, Claim


@dataclass
class FakeNode:
    """Minimal stand-in for a llama_index NodeWithScore."""

    text: str
    metadata: dict


def _nodes():
    return [
        FakeNode(
            text="Chunk zero.",
            metadata={"doc_id": "doc_00001", "chunk_idx_in_doc": 0},
        ),
        FakeNode(
            text="Chunk one.",
            metadata={"doc_id": "doc_00001", "chunk_idx_in_doc": 1},
        ),
    ]


def _good_payload() -> dict:
    return {
        "summary": "Stub answer.",
        "claims": [
            {
                "text": "First claim.",
                "citations": [
                    {"doc_id": "doc_00001", "chunk_idx_in_doc": 0}
                ],
            },
            {
                "text": "Second claim.",
                "citations": [
                    {"doc_id": "doc_00001", "chunk_idx_in_doc": 1}
                ],
            },
        ],
    }


def _good_answer() -> Answer:
    return Answer.model_validate(_good_payload())


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


class TestStructural(unittest.TestCase):
    def test_well_formed_json_parses(self):
        raw = json.dumps(_good_payload())
        answer, violations = validate_structural(raw)
        self.assertIsNotNone(answer)
        self.assertEqual(violations, [])

    def test_fenced_json_parses(self):
        raw = "```json\n" + json.dumps(_good_payload()) + "\n```"
        answer, violations = validate_structural(raw)
        self.assertIsNotNone(answer)
        self.assertEqual(violations, [])

    def test_empty_string_yields_structural_violation(self):
        answer, violations = validate_structural("")
        self.assertIsNone(answer)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].kind, "structural")

    def test_no_json_object_yields_structural_violation(self):
        answer, violations = validate_structural("just some prose")
        self.assertIsNone(answer)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].kind, "structural")
        self.assertIn("No JSON object", violations[0].message)

    def test_malformed_json_yields_structural_violation(self):
        answer, violations = validate_structural('{"summary": "x", "claims": [')
        self.assertIsNone(answer)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].kind, "structural")

    def test_schema_violation_yields_structural_violation(self):
        bad = json.dumps({"summary": "x", "claims": []})
        answer, violations = validate_structural(bad)
        self.assertIsNone(answer)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].kind, "structural")
        self.assertIn("Schema validation failed", violations[0].message)


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


class TestGroundedness(unittest.TestCase):
    def test_all_citations_grounded(self):
        violations = validate_groundedness(_good_answer(), _nodes())
        self.assertEqual(violations, [])

    def test_unknown_doc_id_is_violation(self):
        ans = Answer.model_validate(
            {
                "summary": "x",
                "claims": [
                    {
                        "text": "bogus",
                        "citations": [
                            {"doc_id": "doc_99999", "chunk_idx_in_doc": 0}
                        ],
                    }
                ],
            }
        )
        violations = validate_groundedness(ans, _nodes())
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].kind, "groundedness")
        self.assertEqual(violations[0].location, "claims[0].citations[0]")

    def test_unknown_chunk_idx_is_violation(self):
        ans = Answer.model_validate(
            {
                "summary": "x",
                "claims": [
                    {
                        "text": "bogus",
                        "citations": [
                            {"doc_id": "doc_00001", "chunk_idx_in_doc": 42}
                        ],
                    }
                ],
            }
        )
        violations = validate_groundedness(ans, _nodes())
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].kind, "groundedness")

    def test_legacy_source_metadata_is_recognized(self):
        legacy_nodes = [
            FakeNode(
                text="legacy",
                metadata={"source": "doc_legacy.md", "chunk_idx": 3},
            )
        ]
        ans = Answer.model_validate(
            {
                "summary": "x",
                "claims": [
                    {
                        "text": "ok",
                        "citations": [
                            {"doc_id": "doc_legacy", "chunk_idx_in_doc": 3}
                        ],
                    }
                ],
            }
        )
        violations = validate_groundedness(ans, legacy_nodes)
        self.assertEqual(violations, [])


# ---------------------------------------------------------------------------
# SHACL
# ---------------------------------------------------------------------------


class TestShacl(unittest.TestCase):
    def test_grounded_answer_passes_shacl(self):
        violations = validate_shacl(_good_answer(), _nodes())
        self.assertEqual(violations, [], msg=f"unexpected violations: {violations}")

    def test_ungrounded_citation_fails_shacl(self):
        ans = Answer.model_validate(
            {
                "summary": "x",
                "claims": [
                    {
                        "text": "bogus",
                        "citations": [
                            {"doc_id": "doc_99999", "chunk_idx_in_doc": 0}
                        ],
                    }
                ],
            }
        )
        violations = validate_shacl(ans, _nodes())
        self.assertTrue(len(violations) >= 1)
        self.assertEqual(violations[0].kind, "shacl")

    def test_empty_retrieval_makes_any_citation_fail(self):
        # With no retrieved chunks, every citation must be reported as
        # a violation by the dynamic sh:in constraint.
        violations = validate_shacl(_good_answer(), [])
        self.assertTrue(len(violations) >= 1)
        self.assertEqual(violations[0].kind, "shacl")


# ---------------------------------------------------------------------------
# Top-level validate()
# ---------------------------------------------------------------------------


class TestTopLevelValidate(unittest.TestCase):
    def test_clean_answer_yields_no_violations(self):
        raw = json.dumps(_good_payload())
        violations = validate(raw, _nodes())
        self.assertEqual(violations, [], msg=f"unexpected: {violations}")

    def test_structural_failure_short_circuits(self):
        violations = validate("not json at all", _nodes())
        # Only structural violation(s); groundedness/shacl are skipped
        # because there is no parsed Answer to feed them.
        self.assertTrue(len(violations) >= 1)
        self.assertTrue(all(v.kind == "structural" for v in violations))

    def test_groundedness_and_shacl_both_fire_on_bad_citation(self):
        bad = {
            "summary": "x",
            "claims": [
                {
                    "text": "bogus",
                    "citations": [
                        {"doc_id": "doc_99999", "chunk_idx_in_doc": 0}
                    ],
                }
            ],
        }
        violations = validate(json.dumps(bad), _nodes())
        kinds = {v.kind for v in violations}
        # The same logical problem is intentionally reported by two
        # validators (procedural + declarative); see shacl.py docstring.
        self.assertIn("groundedness", kinds)
        self.assertIn("shacl", kinds)

    def test_violation_is_a_dataclass_with_expected_fields(self):
        v = Violation(kind="x", message="m", location="loc")
        self.assertEqual(v.kind, "x")
        self.assertEqual(v.message, "m")
        self.assertEqual(v.location, "loc")
