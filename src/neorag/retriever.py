import warnings

# VectorStoreIndex: llama_index abstraction that wraps a vector store and an embedding model,
# providing a unified interface for building retrievers and query engines.
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine

# VectorIndexRetriever: performs approximate nearest neighbor (ANN) search against the
# vector store, returning the top-k most similar document chunks for a given query embedding.
from llama_index.core.retrievers import VectorIndexRetriever

# SentenceTransformerRerank: a cross-encoder re-ranker that rescores the initially retrieved
# candidates using a more powerful (but slower) model. This drastically improves precision
# by jointly encoding query + document pairs rather than comparing independent embeddings.
from llama_index.core.postprocessor import SentenceTransformerRerank

# HuggingFaceEmbedding: local embedding model that converts text into dense vector
# representations. Used both at index time and at query time to produce comparable embeddings.
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# QdrantClient: client for the Qdrant vector database. In local mode (path=...) it stores
# vectors on disk without requiring a separate server process.
from qdrant_client import QdrantClient
from .config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K_BASE, TOP_K_FINAL, RERANK_MODEL

# QdrantVectorStore: llama_index adapter that bridges QdrantClient to the llama_index
# vector store interface, enabling transparent storage and retrieval of embeddings.
from llama_index.vector_stores.qdrant import QdrantVectorStore

def _patch_qdrant_client(client):
    """Patch QdrantClient if .search() was removed (qdrant-client >= 1.12).

    Background: starting with qdrant-client >= 1.12, the legacy ``.search()``
    method was removed in favor of the new ``.query_points()`` API (which
    returns a response object exposing ``.points`` instead of a bare list).
    However, the version of llama_index's ``QdrantVectorStore`` pinned in
    this project still calls ``client.search(...)`` internally. To stay
    compatible with both old and new qdrant-client releases without forcing
    a downgrade, we monkey-patch a thin ``search()`` shim onto the client
    that delegates to ``query_points()`` and unwraps the ``.points``
    attribute, mimicking the old return type.
    """
    if not hasattr(client, "search") and hasattr(client, "query_points"):
        def _search(collection_name, query_vector, limit, query_filter=None, **kwargs):
            result = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                **kwargs,
            )
            return result.points

        client.search = _search
    return client


class RetrievalPipeline:
    """Thin wrapper around the two-stage retrieval pipeline.

    Orchestrates:
      Stage 1 – ANN retrieval (VectorIndexRetriever) to fetch TOP_K_BASE candidates.
      Stage 2 – Cross-encoder reranking (SentenceTransformerRerank) down to TOP_K_FINAL.

    Exposes a single ``retrieve()`` method that returns a list of
    ``NodeWithScore`` objects, so callers (CLI, evaluation, future generation
    step) do not have to know about the two-stage internals.

    No LLM is involved; answer synthesis is handled elsewhere.
    """

    def __init__(self, base_retriever, reranker):
        self.base_retriever = base_retriever
        self.reranker = reranker

    def retrieve(self, query):
        """Run the full pipeline for ``query`` and return reranked nodes.

        Parameters
        ----------
        query : str | QueryBundle
            Raw query string or a pre-built llama_index ``QueryBundle``.
        """
        if isinstance(query, QueryBundle):
            query_bundle = query
        else:
            query_bundle = QueryBundle(query)

        nodes = self.base_retriever.retrieve(query_bundle)
        nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        return nodes


def get_retrieval_pipeline(collection_name=None):
    """Build and return a :class:`RetrievalPipeline` for ``collection_name``.

    See :class:`RetrievalPipeline` for pipeline semantics. This is the
    preferred entry point for callers that want to run retrieval without
    juggling the individual pipeline stages themselves.
    """
    if collection_name is None:
        collection_name = COLLECTION_NAME

    # Connect to the on-disk Qdrant instance that was populated during indexing
    client = _patch_qdrant_client(QdrantClient(path=QDRANT_PATH))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name
    )

    # Local embedding model – must be the same model used during indexing so that
    # query vectors live in the same vector space as the stored document vectors.
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    # Reconstruct the index from the existing vector store (no re-indexing needed).
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    # Stage 1 – Base retriever: fast ANN search, over-fetches TOP_K_BASE candidates
    # so the reranker has a rich pool to choose from.
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K_BASE
    )

    # Stage 2 – Cross-encoder reranker: rescores (query, chunk) pairs jointly.
    # Much more accurate than embedding similarity but too slow to run on the
    # full corpus – hence the two-stage design.
    reranker = SentenceTransformerRerank(
        model=RERANK_MODEL,
        top_n=TOP_K_FINAL
    )

    return RetrievalPipeline(base_retriever, reranker)


def get_query_engine(collection_name=None):
    """Deprecated: use :func:`get_retrieval_pipeline` instead.

    Kept as a thin backwards-compatible shim that still returns the raw
    ``(base_retriever, reranker)`` tuple. New code should use
    :func:`get_retrieval_pipeline`, which returns a :class:`RetrievalPipeline`
    wrapper that orchestrates both stages via a single ``.retrieve()`` call.
    """
    warnings.warn(
        "get_query_engine() is deprecated; use get_retrieval_pipeline() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    pipeline = get_retrieval_pipeline(collection_name=collection_name)
    return pipeline.base_retriever, pipeline.reranker
