"""Build a synthetic parent-document corpus over DiscoResearch/germanrag.

This module materialises the provenance substrate required by FF2
(see improvement-plan.md §3 P1.1, §4 step 3). It groups N consecutive
germanrag chunks into synthetic parent documents ``doc_{k:05d}.md``
written to ``data/germanrag_docs_corpus/``, and emits a sidecar
``provenance.jsonl`` carrying per-chunk provenance metadata
(``doc_id``, ``chunk_idx_in_doc``, ``byte_start``, ``byte_end``,
``sha256``, ``germanrag_row_idx``).

The grouping is deterministic: germanrag rows are iterated in their
native dataset order, ``row["contexts"][0]`` (the positive / gold
passage) is taken as the chunk text, duplicates (by exact string
match) are discarded keeping the first occurrence, and the resulting
chunk stream is split into groups of ``N`` (default 50).

This corpus is NOT a claim about real document provenance. It exists
purely so that the groundedness checker and the SHACL shape on cited
``chunk_idx`` have something real to verify against.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterator

# Separator placed between chunks inside a parent document.
# Carries a machine-readable chunk marker for human inspection;
# authoritative offsets live in provenance.jsonl.
CHUNK_SEP_TEMPLATE = "\n\n---<!-- chunk {i} -->\n\n"

DEFAULT_CHUNKS_PER_DOC = 50
DEFAULT_CORPUS_DIR = Path("data/germanrag_docs_corpus")
PROVENANCE_FILENAME = "provenance.jsonl"


def _iter_unique_positive_chunks(dataset) -> Iterator[tuple[int, str]]:
    """Yield ``(germanrag_row_idx, chunk_text)`` for unique positive chunks.

    Takes ``row["contexts"][0]`` as the positive passage, in dataset row
    order, deduplicating by exact string match (first occurrence wins).
    """
    seen: set[str] = set()
    for row_idx, row in enumerate(dataset):
        contexts = row.get("contexts") or []
        if not contexts:
            continue
        chunk = contexts[0]
        if not isinstance(chunk, str) or not chunk:
            continue
        if chunk in seen:
            continue
        seen.add(chunk)
        yield row_idx, chunk


def build_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    chunks_per_doc: int = DEFAULT_CHUNKS_PER_DOC,
    limit_chunks: int | None = None,
    limit_docs: int | None = None,
) -> dict:
    """Build the synthetic parent-document corpus.

    Parameters
    ----------
    corpus_dir
        Target directory. Wiped and recreated on every invocation.
    chunks_per_doc
        Number of chunks grouped into one synthetic parent document.
    limit_chunks
        If given, stop after this many unique chunks have been collected.
    limit_docs
        If given, stop after this many parent documents have been written.

    Returns
    -------
    dict
        Summary statistics (``n_chunks``, ``n_docs``, ``corpus_dir``).
    """
    # Imported lazily so ``neorag`` does not hard-depend on ``datasets``
    # at import time (helpful error if the package is missing).
    try:
        import datasets as _datasets
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The 'datasets' package is required for build_corpus(). "
            "Install it (see requirements.txt)."
        ) from e

    corpus_dir = Path(corpus_dir)
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    dataset = _datasets.load_dataset(
        "DiscoResearch/germanrag", split="train"
    )

    provenance_path = corpus_dir / PROVENANCE_FILENAME
    n_chunks_total = 0
    n_docs_total = 0

    # Buffer of chunks for the currently-being-assembled parent document.
    # Each entry: (germanrag_row_idx, chunk_text)
    buffer: list[tuple[int, str]] = []

    def _flush(doc_k: int, buf: list[tuple[int, str]], out_fp) -> None:
        """Write one parent document + its provenance rows."""
        doc_id = f"doc_{doc_k:05d}"
        doc_path = corpus_dir / f"{doc_id}.md"

        # Assemble the parent-document body and compute byte offsets on
        # the UTF-8-encoded bytes of that body.
        parts: list[str] = []
        sep_bytes_lengths: list[int] = []
        for i, (_row_idx, chunk_text) in enumerate(buf):
            if i > 0:
                sep = CHUNK_SEP_TEMPLATE.format(i=i)
                parts.append(sep)
                sep_bytes_lengths.append(len(sep.encode("utf-8")))
            parts.append(chunk_text)

        body = "".join(parts)
        body_bytes = body.encode("utf-8")

        # Recompute offsets walking the encoded body.
        offset = 0
        prov_rows: list[dict] = []
        for i, (row_idx, chunk_text) in enumerate(buf):
            if i > 0:
                offset += sep_bytes_lengths[i - 1]
            chunk_bytes = chunk_text.encode("utf-8")
            byte_start = offset
            byte_end = offset + len(chunk_bytes)
            offset = byte_end

            sha = hashlib.sha256(chunk_bytes).hexdigest()
            prov_rows.append(
                {
                    "doc_id": doc_id,
                    "chunk_idx_in_doc": i,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "sha256": sha,
                    "germanrag_row_idx": row_idx,
                }
            )

        # Sanity check: the last offset must match the full body length.
        assert offset == len(body_bytes), (
            f"offset mismatch in {doc_id}: {offset} != {len(body_bytes)}"
        )

        doc_path.write_bytes(body_bytes)
        for row in prov_rows:
            out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    with provenance_path.open("w", encoding="utf-8") as out_fp:
        for row_idx, chunk in _iter_unique_positive_chunks(dataset):
            if limit_chunks is not None and n_chunks_total >= limit_chunks:
                break

            buffer.append((row_idx, chunk))
            n_chunks_total += 1

            if len(buffer) >= chunks_per_doc:
                _flush(n_docs_total, buffer, out_fp)
                n_docs_total += 1
                buffer = []
                if limit_docs is not None and n_docs_total >= limit_docs:
                    break

        # Flush the tail (partial final document), unless we already hit
        # the doc limit on an exact boundary.
        if buffer and (limit_docs is None or n_docs_total < limit_docs):
            _flush(n_docs_total, buffer, out_fp)
            n_docs_total += 1

    return {
        "n_chunks": n_chunks_total,
        "n_docs": n_docs_total,
        "corpus_dir": str(corpus_dir),
    }
