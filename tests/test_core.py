import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# the following requires the package to be properly installed in your environment
# import neorag


class TestCore(unittest.TestCase):
    def setUp(self):
        pass

    # ------------------------------------------------------------------
    # 1. OneToOneMapping
    # ------------------------------------------------------------------
    def test_011_onetoone_basic(self):
        """Key-value pairs are stored and reverse-mapped correctly."""
        from neorag.auxiliary import OneToOneMapping

        m = OneToOneMapping({"a": "x", "b": "y"})
        self.assertEqual(m.a["a"], "x")
        self.assertEqual(m.a["b"], "y")
        self.assertEqual(m.b["x"], "a")
        self.assertEqual(m.b["y"], "b")

    def test_012_onetoone_keyword_init(self):
        """OneToOneMapping can be initialized with keyword arguments."""
        from neorag.auxiliary import OneToOneMapping

        m = OneToOneMapping(a="x", b="y")
        self.assertEqual(m.a["a"], "x")
        self.assertEqual(m.b["x"], "a")

    def test_013_onetoone_add_pair(self):
        """add_pair inserts both directions."""
        from neorag.auxiliary import OneToOneMapping

        m = OneToOneMapping({"a": "x"})
        m.add_pair("b", "y")
        self.assertEqual(m.a["b"], "y")
        self.assertEqual(m.b["y"], "b")

    def test_014_onetoone_remove_pair_strict(self):
        """remove_pair removes both directions and raises KeyError strictly."""
        from neorag.auxiliary import OneToOneMapping

        m = OneToOneMapping({"a": "x"})
        m.remove_pair(key_a="a")
        with self.assertRaises(KeyError):
            _ = m.a["a"]
        with self.assertRaises(KeyError):
            _ = m.b["x"]

    def test_015_onetoone_duplicate_key_a_rejected(self):
        """Adding a key_a that already exists raises KeyError."""
        from neorag.auxiliary import OneToOneMapping

        m = OneToOneMapping({"a": "x"})
        with self.assertRaises(KeyError):
            m.add_pair("a", "y")

    def test_016_onetoone_duplicate_key_b_rejected(self):
        """Adding a key_b that already exists raises KeyError."""
        from neorag.auxiliary import OneToOneMapping

        m = OneToOneMapping({"a": "x"})
        with self.assertRaises(KeyError):
            m.add_pair("b", "x")

    # ------------------------------------------------------------------
    # 2. Provenance JSONL parsing
    # ------------------------------------------------------------------
    def test_020_provenance_load_empty_dir(self):
        """_load_provenance returns empty dict when no provenance.jsonl exists."""
        from neorag.loader import _load_provenance

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _load_provenance(Path(tmpdir))
            self.assertEqual(result, {})

    def test_021_provenance_load_valid_file(self):
        """_load_provenance parses a valid provenance.jsonl correctly."""
        from neorag.loader import _load_provenance

        provenance_rows = [
            {"doc_id": "doc_00000", "chunk_idx_in_doc": 0, "sha256": "abc"},
            {"doc_id": "doc_00000", "chunk_idx_in_doc": 1, "sha256": "def"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            prov_path = Path(tmpdir) / "provenance.jsonl"
            prov_path.write_text("\n".join(json.dumps(r) for r in provenance_rows))
            result = _load_provenance(Path(tmpdir))

        self.assertIn("doc_00000", result)
        self.assertEqual(len(result["doc_00000"]), 2)
        # Chunks must be sorted by chunk_idx_in_doc
        self.assertEqual(result["doc_00000"][0]["chunk_idx_in_doc"], 0)
        self.assertEqual(result["doc_00000"][1]["chunk_idx_in_doc"], 1)

    def test_022_provenance_load_multiple_docs(self):
        """_load_provenance handles multiple parent documents."""
        from neorag.loader import _load_provenance

        provenance_rows = [
            {"doc_id": "doc_00000", "chunk_idx_in_doc": 0, "sha256": "aaa"},
            {"doc_id": "doc_00001", "chunk_idx_in_doc": 0, "sha256": "bbb"},
            {"doc_id": "doc_00001", "chunk_idx_in_doc": 1, "sha256": "ccc"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            prov_path = Path(tmpdir) / "provenance.jsonl"
            prov_path.write_text("\n".join(json.dumps(r) for r in provenance_rows))
            result = _load_provenance(Path(tmpdir))

        self.assertEqual(set(result.keys()), {"doc_00000", "doc_00001"})
        self.assertEqual(len(result["doc_00000"]), 1)
        self.assertEqual(len(result["doc_00001"]), 2)

    # ------------------------------------------------------------------
    # 3. Chunk loading (legacy mode, no provenance)
    # ------------------------------------------------------------------
    def test_030_load_chunks_simple(self):
        """load_chunks returns one Document per .md file without provenance."""
        from neorag.loader import load_chunks

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc_000.md").write_text("# Chunk 1\n\nContent here.")
            (Path(tmpdir) / "doc_001.md").write_text("# Chunk 2\n\nMore content.")

            docs = load_chunks(Path(tmpdir))

        self.assertEqual(len(docs), 2)
        all_texts = [d.text for d in docs]
        self.assertTrue(
            any("# Chunk 1" in t for t in all_texts),
            f"'# Chunk 1' not found in any document text: {all_texts}",
        )
        self.assertTrue(
            any("# Chunk 2" in t for t in all_texts),
            f"'# Chunk 2' not found in any document text: {all_texts}",
        )

    def test_031_load_chunks_metadata_no_provenance(self):
        """load_chunks attaches source and chunk_idx metadata (legacy mode)."""
        from neorag.loader import load_chunks

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc_42.md").write_text("Content.")

            docs = load_chunks(Path(tmpdir))

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.metadata["source"], "doc_42.md")
        self.assertEqual(doc.metadata["chunk_idx"], 42)

    def test_032_load_chunks_with_provenance(self):
        """load_chunks emits one Document per chunk when provenance.jsonl exists.

        All content is pure ASCII so byte offsets are exact:
          - "AAA"                     → bytes 0-2   (chunk 0)
          - separator                → bytes 3-25  (23 bytes)
          - "BBB"                     → bytes 26-29 (chunk 1)
        """
        from neorag.loader import load_chunks

        sep = "\n\n---<!-- chunk 1 -->\n\n"
        parent_doc = f"AAA{sep}BBB"  # pure ASCII → 1 byte per char

        # Pre-computed offsets for "AAA" and "BBB" in the above string
        # total length = 29; sep = 23 bytes; "AAA" at 0-2, sep at 3-25, "BBB" at 26-29
        byte_start_0, byte_end_0 = 0, 3   # "AAA"
        byte_start_1, byte_end_1 = 26, 29  # "BBB"

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc_00000.md").write_bytes(parent_doc.encode("utf-8"))

            provenance_rows = [
                {
                    "doc_id": "doc_00000", "chunk_idx_in_doc": 0,
                    "byte_start": byte_start_0, "byte_end": byte_end_0,
                    "sha256": "a" * 64,
                },
                {
                    "doc_id": "doc_00000", "chunk_idx_in_doc": 1,
                    "byte_start": byte_start_1, "byte_end": byte_end_1,
                    "sha256": "b" * 64,
                },
            ]
            prov_path = Path(tmpdir) / "provenance.jsonl"
            prov_path.write_text("\n".join(json.dumps(r) for r in provenance_rows))

            docs = load_chunks(Path(tmpdir))

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].text, "AAA")
        self.assertEqual(docs[1].text, "BBB")
        # Verify provenance metadata is attached
        for doc in docs:
            self.assertIn("doc_id", doc.metadata)
            self.assertIn("sha256", doc.metadata)
            self.assertIn("byte_start", doc.metadata)
            self.assertIn("byte_end", doc.metadata)
            self.assertIn("chunk_idx_in_doc", doc.metadata)

    # ------------------------------------------------------------------
    # 4. validate_dirs
    # ------------------------------------------------------------------
    def test_040_validate_dirs_missing_index_raises(self):
        """validate_dirs raises FileNotFoundError when INDEX_DIR does not exist."""
        from neorag.config import INDEX_DIR as orig_index_dir, validate_dirs

        fake_index = Path("/tmp/neorag_nonexistent_index_dir_12345")
        if fake_index.exists():
            shutil.rmtree(fake_index)

        import neorag.config as cfg_module
        original_val = cfg_module.INDEX_DIR
        cfg_module.INDEX_DIR = fake_index
        try:
            with self.assertRaises(FileNotFoundError) as ctx:
                validate_dirs()
            self.assertIn("neorag --bootstrap", str(ctx.exception))
        finally:
            cfg_module.INDEX_DIR = original_val

    # ------------------------------------------------------------------
    # 5. _patch_qdrant_client
    # ------------------------------------------------------------------
    def test_050_patch_keeps_search_if_present(self):
        """_patch_qdrant_client leaves client.search untouched when it already exists."""
        from neorag.retriever import _patch_qdrant_client

        class FakeClient:
            def search(self, *args, **kwargs):
                return "original_search"

        client = FakeClient()
        result = _patch_qdrant_client(client)
        self.assertIs(result, client)
        self.assertEqual(client.search("any", [], 5), "original_search")

    def test_051_patch_adds_search_when_missing(self):
        """_patch_qdrant_client adds a search() method delegating to query_points().

        Background: see docstring for _patch_qdrant_client for why we need this.
        """
        from neorag.retriever import _patch_qdrant_client

        class FakeClient:
            def __init__(self):
                self.calls = []

            def query_points(self, collection_name, query, limit, **kwargs):
                self.calls.append((collection_name, query, limit))
                class FakePoints:
                    points = ["result_a", "result_b"]
                return FakePoints()

        client = FakeClient()
        result = _patch_qdrant_client(client)
        self.assertIs(result, client)

        # Patch must have added .search
        self.assertTrue(hasattr(client, "search"))
        points = client.search("my_collection", [0.1, 0.2], limit=5)
        self.assertEqual(points, ["result_a", "result_b"])
        self.assertEqual(client.calls, [("my_collection", [0.1, 0.2], 5)])
