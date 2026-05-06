import json
from collections import defaultdict
from pathlib import Path

from llama_index.core import Document

PROVENANCE_FILENAME = "provenance.jsonl"


def _load_provenance(data_dir: Path) -> dict[str, list[dict]]:
    """Return ``{doc_id: [prov_row, ...]}`` sorted by ``chunk_idx_in_doc``.

    Returns an empty dict if no ``provenance.jsonl`` is present.
    """
    prov_path = data_dir / PROVENANCE_FILENAME
    if not prov_path.is_file():
        return {}

    by_doc: dict[str, list[dict]] = defaultdict(list)
    with prov_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_doc[row["doc_id"]].append(row)

    for doc_id, rows in by_doc.items():
        rows.sort(key=lambda r: r["chunk_idx_in_doc"])
    return dict(by_doc)


def _load_chunks_with_provenance(
    data_dir: Path, prov_by_doc: dict[str, list[dict]]
) -> list[Document]:
    """Emit one Document per chunk, attaching full provenance metadata."""
    documents: list[Document] = []
    for chunk_file in sorted(data_dir.glob("*.md")):
        doc_id = chunk_file.stem
        prov_rows = prov_by_doc.get(doc_id)
        if prov_rows is None:
            # No provenance entry for this file: fall back to whole-file
            # loading with legacy metadata only.
            content = chunk_file.read_text(encoding="utf-8")
            documents.append(
                Document(
                    text=content,
                    metadata={"source": chunk_file.name, "chunk_idx": doc_id},
                )
            )
            continue

        body_bytes = chunk_file.read_bytes()
        for row in prov_rows:
            byte_start = row["byte_start"]
            byte_end = row["byte_end"]
            chunk_bytes = body_bytes[byte_start:byte_end]
            chunk_text = chunk_bytes.decode("utf-8")

            chunk_idx_in_doc = row["chunk_idx_in_doc"]
            metadata = {
                # Legacy keys (kept for backwards compatibility).
                "source": chunk_file.name,
                "chunk_idx": chunk_idx_in_doc,
                # New provenance keys.
                "doc_id": row["doc_id"],
                "chunk_idx_in_doc": chunk_idx_in_doc,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "sha256": row["sha256"],
            }
            if "germanrag_row_idx" in row:
                metadata["germanrag_row_idx"] = row["germanrag_row_idx"]

            documents.append(Document(text=chunk_text, metadata=metadata))
    return documents


def load_chunks(data_dir: Path):
    """Load markdown chunks with metadata.

    If ``data_dir`` contains a ``provenance.jsonl`` sidecar (as emitted by
    :mod:`neorag.build_corpus`), one :class:`Document` is emitted per chunk,
    carrying both the legacy ``source`` / ``chunk_idx`` keys and the new
    provenance keys (``doc_id``, ``chunk_idx_in_doc``, ``byte_start``,
    ``byte_end``, ``sha256``, and optionally ``germanrag_row_idx``).

    Otherwise the legacy behaviour is preserved: one Document per ``.md``
    file with only ``source`` and ``chunk_idx`` metadata.
    """
    data_dir = Path(data_dir)
    prov_by_doc = _load_provenance(data_dir)
    if prov_by_doc:
        return _load_chunks_with_provenance(data_dir, prov_by_doc)

    documents = []
    for chunk_file in data_dir.glob("*.md"):
        content = chunk_file.read_text(encoding="utf-8")

        # Try to extract numeric index from filename, fallback to full stem if fails
        try:
            chunk_idx = int(chunk_file.stem.split("_")[-1])
        except (ValueError, IndexError):
            # If extraction fails, use the full filename stem as string identifier
            chunk_idx = chunk_file.stem

        doc = Document(
            text=content,
            metadata={
                "source": chunk_file.name,
                "chunk_idx": chunk_idx
            }
        )
        documents.append(doc)
    return documents
