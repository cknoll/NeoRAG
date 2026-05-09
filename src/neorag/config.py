import os
from pathlib import Path

# Paths
DATA_DIR = Path("data")
INDEX_DIR = Path("index")

# LLM backend (see src/neorag/llm_client.py).
# Defaults match the improvement plan: OpenRouter as the default remote
# backend, with model/base-url overridable here.
LLM_PROVIDER = "openrouter"  # one of: "openai", "openrouter", "anthropic", "ollama", "stub"
LLM_MODEL = "google/gemini-2.0-flash-001"
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT_S = 60.0

# API keys are read from a TOML file (shippable example: config-example.toml).
# Override via env var NEORAG_CONFIG_TOML if needed.
CONFIG_TOML_PATH = Path(os.environ.get("NEORAG_CONFIG_TOML", "config.toml"))
# Mapping from provider name to the TOML field that holds the API key.
LLM_API_KEY_TOML_FIELD = {
    "openrouter": "openrouter_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
}




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
