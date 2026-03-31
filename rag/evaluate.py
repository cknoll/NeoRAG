import json
from pathlib import Path
from rag.retriever import get_query_engine

def evaluate(test_path: Path):
    """Evaluate retrieval accuracy on test set."""
    test_data = json.loads(test_path.read_text())
    query_engine = get_query_engine()
    
    mrr_total = 0
    recall_at_5 = 0
    
    for item in test_data:
        query = item["query"]
        expected_sources = set(item["expected_sources"])
        
        response = query_engine.query(query)
        retrieved_sources = [node.metadata["source"] for node in response.source_nodes]
        
        # MRR@10
        for rank, source in enumerate(retrieved_sources[:10], 1):
            if source in expected_sources:
                mrr_total += 1 / rank
                break
        
        # Recall@5
        if any(source in expected_sources for source in retrieved_sources[:5]):
            recall_at_5 += 1
    
    print(f"MRR@10: {mrr_total / len(test_data):.3f}")
    print(f"Recall@5: {recall_at_5 / len(test_data):.3f}")
