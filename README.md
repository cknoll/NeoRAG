[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# RAG System with High-Accuracy Retrieval

This is a RAG (Retrieval-Augmented Generation) system that uses local open-source models for embeddings and re-ranking to achieve high retrieval accuracy.

## Project Structure

```
neorag/
├── data/                 # Your markdown chunk files
├── index/                # Qdrant vector DB storage
├── src/
│   └── neorag/
│       ├── __init__.py
│       ├── config.py         # Model & path configurations
│       ├── loader.py         # Chunk loading with metadata
│       ├── indexer.py        # Build vector index
│       ├── retriever.py      # Core retrieval with re-ranking
│       ├── evaluate.py       # Accuracy evaluation
│       └── cli.py            # CLI interface (entry point)
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
python -m neorag.evaluate eval/test_set.json
```

## Configuration

Key settings in `neorag/config.py`:

- `EMBEDDING_MODEL`: "BAAI/bge-base-en-v1.5" (excellent balance of quality/speed)
- `RERANK_MODEL`: "cross-encoder/ms-marco-MiniLM-L-6-v2" (critical for accuracy)
- `TOP_K_BASE`: 20 (retrieve more for re-ranking)
- `TOP_K_FINAL`: 3 (final results after re-ranking)

## Synthetic Parent-Document Corpus (Provenance Substrate)

NeoRAG ships a helper that materialises a *synthetic* parent-document
corpus from [`DiscoResearch/germanrag`](https://huggingface.co/datasets/DiscoResearch/germanrag).
The germanrag dataset is distributed pre-chunked, with no source
documents, so there is nothing meaningful to record byte offsets
*into*. The `build-corpus` command groups N consecutive germanrag
chunks (default: 50) into synthetic parent documents
`doc_{k:05d}.md` under `data/sample_corpus/`, and writes a sidecar
`provenance.jsonl` carrying, per chunk:

- `doc_id`, `chunk_idx_in_doc`
- `byte_start`, `byte_end` (offsets into the UTF-8 bytes of the parent `.md`)
- `sha256` of the chunk bytes
- `germanrag_row_idx` (original dataset row for traceability)

### Why it exists

This corpus is the **provenance substrate** for the neurosymbolic
validator (FF2): it gives the groundedness checker and the SHACL
shape on cited `chunk_idx` something real to verify against.
It is **not** a claim about real document provenance — the parent
documents are synthetic groupings.

### Build it

```bash
neorag build-corpus
# or, to cap the output for fast iteration:
neorag build-corpus --limit-docs 10
```

Then index the corpus as usual:

```bash
neorag index --data-dir data/sample_corpus
```

The loader automatically detects `data/sample_corpus/provenance.jsonl`
and attaches the provenance keys (`doc_id`, `chunk_idx_in_doc`,
`byte_start`, `byte_end`, `sha256`) to each chunk's metadata, while
preserving the legacy `source` / `chunk_idx` keys.

## Evaluate with GermanRAG Dataset

### Prerequisites

1. Create a `config.toml` in the project root with your OpenRouter API key:
   ```toml
   openrouter_api_key = "sk-or-..."
   ```
2. Install additional dependencies:
   ```bash
   pip install ragas datasets langchain-openai
   ```

### Run the evaluation

Run `interactive-testing.py` interactively (e.g. in VS Code) or as a script:

```bash
python interactive-testing.py
```

The script does:
- Load the GermanRAG dataset from Hugging Face.
- Extract all unique context passages and index them into a **separate** Qdrant collection (`germanrag_chunks`), leaving default index untouched.
- Evaluate the retrieval results with Ragas metrics.
