"""Tests for the Pydantic answer schema and structured generation
(improvement-plan2 step 2)."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from pydantic import ValidationError

from neorag.generate import (
    StructuredGenerationResult,
    build_structured_prompt,
    generate_structured_answer,
)
from neorag.llm_client import StubBackend
from neorag.validate.schema import Answer, Citation, Claim


@dataclass
class FakeNode:
    """Minimal stand-in for a llama_index NodeWithScore."""

    text: str
    metadata: dict


class TestCitation(unittest.TestCase):
    def test_minimal_citation_accepted(self):
        c = Citation(doc_id="doc_00001", chunk_idx_in_doc=0)
        self.assertEqual(c.doc_id, "doc_00001")
        self.assertEqual(c.chunk_idx_in_doc, 0)
        self.assertIsNone(c.quote)

    def test_citation_with_quote(self):
        c = Citation(doc_id="doc_00001", chunk_idx_in_doc=2, quote="snippet")
        self.assertEqual(c.quote, "snippet")

    def test_negative_chunk_idx_rejected(self):
        with self.assertRaises(ValidationError):
            Citation(doc_id="doc_00001", chunk_idx_in_doc=-1)

    def test_empty_doc_id_rejected(self):
        with self.assertRaises(ValidationError):
            Citation(doc_id="", chunk_idx_in_doc=0)

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValidationError):
            Citation(doc_id="d", chunk_idx_in_doc=0, bogus="x")


class TestClaim(unittest.TestCase):
    def test_claim_requires_at_least_one_citation(self):
        with self.assertRaises(ValidationError):
            Claim(text="some claim", citations=[])

    def test_claim_text_must_be_nonempty(self):
        with self.assertRaises(ValidationError):
            Claim(
                text="",
                citations=[Citation(doc_id="d", chunk_idx_in_doc=0)],
            )

    def test_valid_claim_accepted(self):
        claim = Claim(
            text="The sky is blue.",
            citations=[Citation(doc_id="d", chunk_idx_in_doc=0)],
        )
        self.assertEqual(len(claim.citations), 1)


class TestAnswer(unittest.TestCase):
    def _valid_payload(self) -> dict:
        return {
            "summary": "The sky is blue because of Rayleigh scattering.",
            "claims": [
                {
                    "text": "The sky appears blue.",
                    "citations": [
                        {"doc_id": "doc_00001", "chunk_idx_in_doc": 0}
                    ],
                },
                {
                    "text": "This is caused by Rayleigh scattering.",
                    "citations": [
                        {
                            "doc_id": "doc_00001",
                            "chunk_idx_in_doc": 1,
                            "quote": "Rayleigh scattering",
                        }
                    ],
                },
            ],
        }

    def test_valid_answer_round_trips_through_json(self):
        ans = Answer.model_validate(self._valid_payload())
        as_json = ans.model_dump_json()
        ans2 = Answer.model_validate_json(as_json)
        self.assertEqual(ans, ans2)

    def test_answer_requires_at_least_one_claim(self):
        payload = self._valid_payload()
        payload["claims"] = []
        with self.assertRaises(ValidationError):
            Answer.model_validate(payload)

    def test_answer_requires_summary(self):
        payload = self._valid_payload()
        del payload["summary"]
        with self.assertRaises(ValidationError):
            Answer.model_validate(payload)

    def test_empty_summary_rejected(self):
        payload = self._valid_payload()
        payload["summary"] = ""
        with self.assertRaises(ValidationError):
            Answer.model_validate(payload)


class TestBuildStructuredPrompt(unittest.TestCase):
    def _nodes(self):
        return [
            FakeNode(
                text="Chunk one text.",
                metadata={"doc_id": "doc_00001", "chunk_idx_in_doc": 0},
            ),
            FakeNode(
                text="Chunk two text.",
                metadata={"doc_id": "doc_00001", "chunk_idx_in_doc": 1},
            ),
        ]

    def test_two_messages_system_then_user(self):
        msgs = build_structured_prompt("What is X?", self._nodes())
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")

    def test_system_prompt_mentions_json_and_schema(self):
        msgs = build_structured_prompt("What is X?", self._nodes())
        sys_content = msgs[0]["content"]
        self.assertIn("JSON", sys_content)
        self.assertIn("summary", sys_content)
        self.assertIn("claims", sys_content)
        self.assertIn("citations", sys_content)
        self.assertIn("doc_id", sys_content)
        self.assertIn("chunk_idx_in_doc", sys_content)

    def test_user_message_contains_chunk_ids_and_query(self):
        msgs = build_structured_prompt("What is X?", self._nodes())
        user_content = msgs[1]["content"]
        self.assertIn("[doc_00001#0]", user_content)
        self.assertIn("[doc_00001#1]", user_content)
        self.assertIn("What is X?", user_content)


class TestGenerateStructuredAnswer(unittest.TestCase):
    def _nodes(self):
        return [
            FakeNode(
                text="Some chunk.",
                metadata={"doc_id": "doc_00001", "chunk_idx_in_doc": 0},
            )
        ]

    def _canned(self, payload: dict) -> str:
        return json.dumps(payload)

    def _good_payload(self) -> dict:
        return {
            "summary": "Stub answer.",
            "claims": [
                {
                    "text": "Stub claim.",
                    "citations": [
                        {"doc_id": "doc_00001", "chunk_idx_in_doc": 0}
                    ],
                }
            ],
        }

    def test_well_formed_json_parses_into_answer(self):
        backend = StubBackend(canned_response=self._canned(self._good_payload()))
        result = generate_structured_answer("Q?", self._nodes(), backend)

        self.assertIsInstance(result, StructuredGenerationResult)
        self.assertIsNone(result.parse_error)
        self.assertIsInstance(result.parsed, Answer)
        self.assertEqual(result.parsed.summary, "Stub answer.")
        self.assertEqual(len(result.parsed.claims), 1)

    def test_json_in_markdown_fence_still_parses(self):
        body = self._canned(self._good_payload())
        backend = StubBackend(canned_response=f"```json\n{body}\n```")
        result = generate_structured_answer("Q?", self._nodes(), backend)
        self.assertIsNone(result.parse_error)
        self.assertIsNotNone(result.parsed)

    def test_invalid_json_surfaces_parse_error(self):
        backend = StubBackend(canned_response="this is not JSON at all")
        result = generate_structured_answer("Q?", self._nodes(), backend)
        self.assertIsNone(result.parsed)
        self.assertIsNotNone(result.parse_error)
        self.assertIn("No JSON object", result.parse_error)

    def test_malformed_json_surfaces_parse_error(self):
        backend = StubBackend(canned_response='{"summary": "x", "claims": [')
        result = generate_structured_answer("Q?", self._nodes(), backend)
        self.assertIsNone(result.parsed)
        self.assertIsNotNone(result.parse_error)

    def test_schema_violation_surfaces_parse_error(self):
        bad = {"summary": "x", "claims": []}  # empty claims list
        backend = StubBackend(canned_response=self._canned(bad))
        result = generate_structured_answer("Q?", self._nodes(), backend)
        self.assertIsNone(result.parsed)
        self.assertIsNotNone(result.parse_error)
        self.assertIn("Schema validation failed", result.parse_error)

    def test_raw_text_is_preserved_verbatim(self):
        raw = "garbage in, garbage out"
        backend = StubBackend(canned_response=raw)
        result = generate_structured_answer("Q?", self._nodes(), backend)
        self.assertEqual(result.raw_text, raw)

    def test_backend_receives_structured_prompt(self):
        backend = StubBackend(canned_response=self._canned(self._good_payload()))
        generate_structured_answer("What is X?", self._nodes(), backend)
        self.assertIsNotNone(backend.last_messages)
        self.assertEqual(backend.last_messages[0]["role"], "system")
        self.assertIn("JSON", backend.last_messages[0]["content"])
        self.assertIn("[doc_00001#0]", backend.last_messages[1]["content"])
