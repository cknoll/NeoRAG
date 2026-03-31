# RAG System with High-Accuracy Retrieval

This is a RAG (Retrieval-Augmented Generation) system that uses local open-source models for embeddings and re-ranking to achieve high retrieval accuracy.

## Project Structure

```
rag/
├── data/                 # Your markdown chunk files
├── index/                # Qdrant vector DB storage
├── rag/
│   ├── __init__.py
│   ├── config.py         # Model & path configurations
│   ├── loader.py         # Chunk loading with metadata
│   ├── indexer.py        # Build vector index
│   ├── retriever.py      # Core retrieval with re-ranking
│   └── evaluate.py       # Accuracy evaluation
├── cli.py                # CLI interface
├── requirements.txt
└── .env                  # API keys (if needed)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Build Index

Place your markdown chunk files in the `data/` directory, then run:

```bash
python cli.py index --data-dir data/
```

### Query

```bash
python cli.py query "your query here"
```

### Evaluate

Create a test set in `eval/test_set.json` and run:

```bash
python -m rag.evaluate eval/test_set.json
```

## Configuration

Key settings in `rag/config.py`:

- `EMBEDDING_MODEL`: "BAAI/bge-base-en-v1.5" (excellent balance of quality/speed)
- `RERANK_MODEL`: "cross-encoder/ms-marco-MiniLM-L-6-v2" (critical for accuracy)
- `TOP_K_BASE`: 20 (retrieve more for re-ranking)
- `TOP_K_FINAL`: 3 (final results after re-ranking)

## Accuracy Improvement Tips

1. **Verify chunk quality**: Chunks should be 300-500 tokens, coherent
2. **Add metadata**: Include document titles, section headers if available
3. **Tune TOP_K_BASE**: Increase to 30-50 if recall is low
4. **Try bge-large**: If you have GPU, use `BAAI/bge-large-en-v1.5` for better embeddings
5. **Hybrid search**: Enable Qdrant's hybrid mode for keyword + semantic
6. **Query expansion**: Add HyDE postprocessor for complex queries
