from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.postprocessor.sentence_transformers_rerank import SentenceTransformerRerank
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K_BASE, TOP_K_FINAL, RERANK_MODEL
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

def get_query_engine():
    """Create retrieval engine with re-ranking."""
    # Reuse existing index
    client = QdrantClient(path=QDRANT_PATH)
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

    # Assemble query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=base_retriever,
        node_postprocessors=[reranker]
    )

    return query_engine
