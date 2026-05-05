from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from .config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL

def build_index(documents, collection_name=None):
    """Build vector index with Qdrant.

    Parameters
    ----------
    documents : list
        Documents to index.
    collection_name : str, optional
        Qdrant collection name. Defaults to COLLECTION_NAME from config.
    """
    if collection_name is None:
        collection_name = COLLECTION_NAME
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Use high-quality local embedding model
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    return index
