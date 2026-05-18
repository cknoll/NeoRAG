"""End-to-end retrieve→generate tests (improvement-plan v2, step 1).

Two flavours:

* a fast test that stubs the retriever and only exercises generate_answer
  against StubBackend — always runs;
* a slow integration test that builds a tiny real index from a handcrafted
  corpus and runs the full RetrievalPipeline + StubBackend end-to-end —
  marked ``slow`` and skipped unless ``--run-slow`` is passed to pytest.
"""

import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from neorag.generate import generate_answer
from neorag.llm_client import StubBackend


@dataclass
class FakeNode:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class TestPipelineE2EFast(unittest.TestCase):
    """Stubbed-retriever end-to-end test (always on)."""

    def test_retrieve_then_generate_passes_chunk_ids_to_backend(self):
        # Pretend the retriever returned these two chunks.
        retrieved = [
            FakeNode(
                text="Berlin is the capital of Germany.",
                metadata={"doc_id": "doc_00042", "chunk_idx_in_doc": 0},
                score=0.9,
            ),
            FakeNode(
                text="Munich is in Bavaria.",
                metadata={"doc_id": "doc_00042", "chunk_idx_in_doc": 1},
                score=0.7,
            ),
        ]
        backend = StubBackend(canned_response="Berlin.")
        answer = generate_answer(
            "What is the capital of Germany?", retrieved, backend
        )

        self.assertEqual(answer, "Berlin.")
        self.assertEqual(backend.call_count, 1)
        user_msg = backend.last_messages[1]["content"]
        self.assertIn("[doc_00042#0]", user_msg)
        self.assertIn("[doc_00042#1]", user_msg)


@pytest.mark.slow
class TestPipelineE2ESlow(unittest.TestCase):
    """Real-retriever end-to-end test (skipped by default; opt in via --run-slow).

    Builds a tiny on-disk index from a handcrafted corpus, runs the real
    two-stage retrieval pipeline, then drives generate_answer against
    StubBackend. Pulls in HuggingFace + Qdrant on first run.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp(prefix="neorag-e2e-"))
        cls._corpus = cls._tmp / "corpus"
        cls._corpus.mkdir()
        # Two tiny markdown "documents".
        (cls._corpus / "doc_00000.md").write_text(
            "Berlin is the capital of Germany. The Spree river flows through it.",
            encoding="utf-8",
        )
        (cls._corpus / "doc_00001.md").write_text(
            "Paris is the capital of France. The Seine river flows through it.",
            encoding="utf-8",
        )
        cls._index_dir = cls._tmp / "index"
        cls._index_dir.mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_real_retrieve_then_stub_generate(self):
        from neorag import config as cfg
        from neorag.indexer import build_index
        from neorag.loader import load_chunks
        from neorag.retriever import get_retrieval_pipeline

        collection = "test_e2e_collection"

        # Redirect QDRANT_PATH to a throwaway directory for the duration
        # of this test so we don't clobber a developer's real index.
        qdrant_path = str(self._index_dir / "qdrant")
        with mock.patch.object(cfg, "QDRANT_PATH", qdrant_path), mock.patch(
            "neorag.indexer.QDRANT_PATH", qdrant_path
        ), mock.patch("neorag.retriever.QDRANT_PATH", qdrant_path):
            documents = load_chunks(self._corpus)
            self.assertGreater(len(documents), 0)

            build_index(documents, collection_name=collection)
            pipeline = get_retrieval_pipeline(collection_name=collection)
            nodes = pipeline.retrieve("What is the capital of Germany?")

        self.assertGreater(len(nodes), 0)

        backend = StubBackend(canned_response="Berlin.")
        answer = generate_answer(
            "What is the capital of Germany?", nodes, backend
        )
        self.assertEqual(answer, "Berlin.")
        self.assertEqual(backend.call_count, 1)


if __name__ == "__main__":
    unittest.main()
