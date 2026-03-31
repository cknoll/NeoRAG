from pathlib import Path
from llama_index.core import Document

def load_chunks(data_dir: Path):
    """Load markdown chunks with metadata."""
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
