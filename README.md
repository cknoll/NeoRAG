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

## Evaluate with GermanRAG Dataset

The `tmp.py` script evaluates the RAG system against the [DiscoResearch/germanrag](https://huggingface.co/datasets/DiscoResearch/germanrag) dataset using [Ragas](https://docs.ragas.io/) metrics.

### Prerequisites

1. **Build the index first** with your own data (see *Build Index* above).
2. Create a `config.toml` in the project root with your OpenRouter API key:
   ```toml
   openrouter_api_key = "sk-or-..."
   ```
3. Install additional dependencies:
   ```bash
   pip install ragas datasets langchain-openai
   ```

### Run the evaluation

```bash
python tmp.py
```

The script will:
1. Load the GermanRAG dataset from Hugging Face.
2. For each test question, run the full two-stage retrieval pipeline (ANN search → cross-encoder reranking) to find relevant chunks.
3. Use the configured LLM (via OpenRouter) to generate an answer from the retrieved context.
4. Evaluate the results with Ragas metrics (e.g. `context_precision`).

## Accuracy Improvement Tips

1. **Verify chunk quality**: Chunks should be 300-500 tokens, coherent
2. **Add metadata**: Include document titles, section headers if available
3. **Tune TOP_K_BASE**: Increase to 30-50 if recall is low
4. **Try bge-large**: If you have GPU, use `BAAI/bge-large-en-v1.5` for better embeddings
5. **Hybrid search**: Enable Qdrant's hybrid mode for keyword + semantic
6. **Query expansion**: Add HyDE postprocessor for complex queries
