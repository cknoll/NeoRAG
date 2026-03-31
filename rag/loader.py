from pathlib import Path
from llama_index.core import Document

def load_chunks(data_dir: Path):
    """Load markdown chunks with metadata."""
    documents = []
    for chunk_file in data_dir.glob("*.md"):
        content = chunk_file.read_text(encoding="utf-8")
        doc = Document(
            text=content,
            metadata={
                "source": chunk_file.name,
                "chunk_idx": int(chunk_file.stem.split("_")[-1])  # Assumes filename_001.md
            }
        )
        documents.append(doc)
    return documents
