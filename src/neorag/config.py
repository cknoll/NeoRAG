import os
from pathlib import Path

# Paths
DATA_DIR = Path("data")
INDEX_DIR = Path("index")


def ensure_dirs():
    """Create all required directories for NeoRAG (called by --bootstrap)."""
    INDEX_DIR.mkdir(exist_ok=True)
    # Future directory creation can be added here


def validate_dirs():
    """Validate that required directories exist; raise error if not.

    The error message instructs the user to use the --bootstrap option to
    create missing directories.
    """
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Required index directory '{INDEX_DIR}' does not exist. "
            f"Run 'neorag --bootstrap' to create it."
        )

# Models (open-source, local)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Excellent balance of quality/speed
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Critical for accuracy

# Qdrant
QDRANT_PATH = str(INDEX_DIR / "qdrant")
COLLECTION_NAME = "rag_chunks"

# Retrieval
TOP_K_BASE = 20  # Retrieve more for re-ranking
TOP_K_FINAL = 3  # Final results after re-ranking
