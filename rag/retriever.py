# VectorStoreIndex: llama_index abstraction that wraps a vector store and an embedding model,
# providing a unified interface for building retrievers and query engines.
from llama_index.core import VectorStoreIndex
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
    """Patch QdrantClient if .search() was removed (qdrant-client >= 1.12)."""
    if not hasattr(client, "search") and hasattr(client, "query_points"):
        from qdrant_client.models import models

        def _search(collection_name, query_vector, limit, query_filter=None, **kwargs):
            from qdrant_client.models import models as _models
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


def get_query_engine(collection_name=None):
    """Create and return a two-stage retrieval pipeline (retriever + reranker).

    Stage 1 – Retriever: embeds the query and fetches TOP_K_BASE candidate chunks
    from Qdrant via fast ANN search (high recall, moderate precision).

    Stage 2 – Reranker: a cross-encoder rescores each candidate against the query,
    keeping only TOP_K_FINAL results (high precision).

    Parameters
    ----------
    collection_name : str, optional
        Qdrant collection to query. Defaults to COLLECTION_NAME from config.

    Returns a (base_retriever, reranker) tuple so callers can run retrieval
    without requiring an LLM for answer synthesis.
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

    # Stage 1 – Base retriever: performs fast ANN search and returns TOP_K_BASE
    # candidate chunks. We intentionally over-fetch so the reranker has a rich
    # pool of candidates to choose from.
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K_BASE
    )

    # Stage 2 – Cross-encoder reranker: takes each (query, chunk) pair and produces
    # a relevance score using a cross-encoder model. Much more accurate than
    # embedding similarity alone, but too slow to run on the full corpus – hence
    # the two-stage design. Keeps only the TOP_K_FINAL best results.
    reranker = SentenceTransformerRerank(
        model=RERANK_MODEL,
        top_n=TOP_K_FINAL
    )

    # Return both components separately – the caller orchestrates the pipeline
    # without needing an LLM for answer synthesis.
    return base_retriever, reranker
