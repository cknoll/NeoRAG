import os
from pathlib import Path

# Paths
DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

# Models (open-source, local)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Excellent balance of quality/speed
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Critical for accuracy

# Qdrant
QDRANT_PATH = str(INDEX_DIR / "qdrant")
COLLECTION_NAME = "rag_chunks"

# Retrieval
TOP_K_BASE = 20  # Retrieve more for re-ranking
TOP_K_FINAL = 3  # Final results after re-ranking
