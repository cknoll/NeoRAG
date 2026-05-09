"""Tests for neorag.generate (improvement-plan v2, step 1)."""

import unittest
from dataclasses import dataclass, field
from typing import Any, Dict

from neorag.generate import (
    SYSTEM_PROMPT,
    _chunk_id,
    build_prompt,
    generate_answer,
    render_context,
)
from neorag.llm_client import StubBackend


@dataclass
class FakeNode:
    """Minimal stand-in for a llama_index NodeWithScore.

    Exposes ``.text`` and ``.metadata`` directly, which is the interface
    ``neorag.generate`` relies on.
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class TestChunkId(unittest.TestCase):
    """Does _chunk_id render the expected [doc_id#chunk_idx_in_doc] form?"""

    def test_full_provenance_metadata(self):
        node = FakeNode(
            text="hello",
            metadata={"doc_id": "doc_00001", "chunk_idx_in_doc": 3},
        )
        self.assertEqual(_chunk_id(node), "[doc_00001#3]")

    def test_strips_md_suffix_from_source_fallback(self):
        node = FakeNode(
            text="hello",
            metadata={"source": "doc_00007.md", "chunk_idx": 2},
        )
        self.assertEqual(_chunk_id(node), "[doc_00007#2]")

    def test_only_source_no_chunk_idx(self):
        node = FakeNode(text="hello", metadata={"source": "notes.md"})
        self.assertEqual(_chunk_id(node), "[notes]")

    def test_no_metadata_at_all(self):
        node = FakeNode(text="hello", metadata={})
        self.assertEqual(_chunk_id(node), "[unknown]")


class TestBuildPrompt(unittest.TestCase):
    """Does build_prompt produce a well-shaped messages list?"""

    def _nodes(self):
        return [
            FakeNode(
                text="Berlin is the capital of Germany.",
                metadata={"doc_id": "doc_00000", "chunk_idx_in_doc": 0},
            ),
            FakeNode(
                text="Paris is the capital of France.",
                metadata={"doc_id": "doc_00000", "chunk_idx_in_doc": 1},
            ),
        ]

    def test_returns_two_messages_system_then_user(self):
        msgs = build_prompt("What is the capital of Germany?", self._nodes())
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(msgs[1]["role"], "user")

    def test_user_message_contains_chunk_ids_and_query(self):
        query = "What is the capital of Germany?"
        msgs = build_prompt(query, self._nodes())
        user = msgs[1]["content"]
        self.assertIn("[doc_00000#0]", user)
        self.assertIn("[doc_00000#1]", user)
        self.assertIn("Berlin is the capital of Germany.", user)
        self.assertIn(query, user)

    def test_empty_context_still_includes_query(self):
        msgs = build_prompt("Hello?", [])
        self.assertEqual(len(msgs), 2)
        self.assertIn("Hello?", msgs[1]["content"])

    def test_render_context_separates_chunks_with_blank_line(self):
        rendered = render_context(self._nodes())
        # Two chunks → exactly one blank-line separator between them.
        self.assertEqual(rendered.count("\n\n"), 1)
        self.assertTrue(rendered.startswith("[doc_00000#0]"))


class TestGenerateAnswer(unittest.TestCase):
    """Does generate_answer drive the LLMBackend correctly?"""

    def test_calls_backend_with_built_prompt_and_returns_content(self):
        nodes = [
            FakeNode(
                text="The Rhine flows through Germany.",
                metadata={"doc_id": "doc_00001", "chunk_idx_in_doc": 0},
            ),
        ]
        backend = StubBackend(canned_response="The Rhine.")
        out = generate_answer("Which river?", nodes, backend)

        self.assertEqual(out, "The Rhine.")
        self.assertEqual(backend.call_count, 1)
        self.assertIsNotNone(backend.last_messages)
        # Built prompt must reach the backend with the chunk id embedded.
        self.assertEqual(backend.last_messages[0]["role"], "system")
        user_msg = backend.last_messages[1]["content"]
        self.assertIn("[doc_00001#0]", user_msg)
        self.assertIn("Which river?", user_msg)


if __name__ == "__main__":
    unittest.main()
