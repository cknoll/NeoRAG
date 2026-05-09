import os
import tomllib
from pathlib import Path
from typing import Optional

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


def load_api_key_from_toml(
    provider_name: str,
    toml_path: Optional[Path] = None,
) -> Optional[str]:
    """Return the API key for ``provider_name`` read from a TOML config file.

    Returns ``None`` if no key is configured (e.g. for ``ollama`` or
    ``stub`` providers, or if the TOML file does not exist). Raises
    :class:`LLMClientError` if the TOML file exists but is malformed.
    """
    field_name = LLM_API_KEY_TOML_FIELD.get(provider_name)
    if field_name is None:
        return None

    path = Path(toml_path) if toml_path is not None else CONFIG_TOML_PATH
    if not path.is_file():
        return None

    try:
        with path.open("rb") as fp:
            data = tomllib.load(fp)
    except (OSError, tomllib.TOMLDecodeError) as e:
        from .llm_client import LLMClientError

        raise LLMClientError(f"Failed to read API key from {path}: {e}") from e

    return data.get(field_name)




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
