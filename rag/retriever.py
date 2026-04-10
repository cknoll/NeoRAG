from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.postprocessor.sentence_transformers_rerank import SentenceTransformerRerank
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from .config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K_BASE, TOP_K_FINAL, RERANK_MODEL
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


def get_query_engine():
    """Create retrieval engine with re-ranking."""
    # Reuse existing index
    client = _patch_qdrant_client(QdrantClient(path=QDRANT_PATH))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME
    )

    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    # Base retriever: get many candidates
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K_BASE
    )

    # Re-ranker: crucial for accuracy
    reranker = SentenceTransformerRerank(
        model=RERANK_MODEL,
        top_n=TOP_K_FINAL
    )

    import os
    os.environ["OPENAI_API_KEY"] = "dummy"

    query_engine = RetrieverQueryEngine.from_args(
        retriever=base_retriever,
        node_postprocessors=[reranker],
        llm=None  # Disable LLM since we only need retrieval
    )

    return query_engine
