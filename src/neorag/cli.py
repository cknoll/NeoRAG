import argparse
from pathlib import Path

# NOTE: Heavy imports (loader, indexer, retriever, build_corpus) are intentionally
# deferred to inside the command handlers so that `--help` and argument parsing
# stay fast. Top-level imports here must remain lightweight.


def _derive_collection_from_data_dir(data_dir: str) -> str:
    """Derive the Qdrant collection name from a corpus directory name.

    Convention: hyphens → underscores; the collection name is the directory
    name after this normalisation. Example: ``data/germanrag_docs_corpus``
    → ``germanrag_docs_corpus``.
    """
    return Path(data_dir).name.replace("-", "_")


def _cmd_index(args):
    """Build vector index from chunks."""
    from .config import CORPUS_DEFAULTS, validate_dirs
    from .loader import load_chunks
    from .indexer import build_index

    validate_dirs()

    # Resolve collection name: explicit override → convention → error.
    if args.collection:
        collection_name = args.collection
    else:
        data_dir_name = Path(args.data_dir).name
        if data_dir_name in CORPUS_DEFAULTS:
            collection_name = CORPUS_DEFAULTS[data_dir_name]
        else:
            collection_name = _derive_collection_from_data_dir(args.data_dir)
            print(
                f"NOTE: no corpus default for '{data_dir_name}'; "
                f"indexing into collection '{collection_name}'"
            )

    print(f"Loading chunks from {args.data_dir}...")
    documents = load_chunks(Path(args.data_dir))
    print(f"Loaded {len(documents)} documents")

    print(f"Building index into collection '{collection_name}'...")
    build_index(documents, collection_name=collection_name)
    print("Index built successfully!")


def _cmd_build_corpus(args):
    """Build the synthetic parent-document corpus over DiscoResearch/germanrag."""
    from .build_corpus import build_corpus

    print(f"Building synthetic corpus in {args.data_dir} ...")
    summary = build_corpus(
        corpus_dir=Path(args.data_dir),
        chunks_per_doc=args.chunks_per_doc,
        limit_chunks=args.limit_chunks,
        limit_docs=args.limit_docs,
    )
    print(
        f"Wrote {summary['n_docs']} parent documents "
        f"({summary['n_chunks']} chunks) to {summary['corpus_dir']}"
    )


def _cmd_query(args):
    """Query the RAG system.

    Runs a two-stage retrieval pipeline:
    1. ANN search in Qdrant to get initial candidate chunks.
    2. Cross-encoder reranking to select the most relevant results.

    By default also runs an LLM generation step over the retrieved
    context. Pass ``--no-generate`` to skip generation and only print
    the retrieved chunks (useful for retrieval debugging).
    """
    from .config import DEFAULT_COLLECTION, validate_dirs
    from .generate import _chunk_id
    from .retriever import get_retrieval_pipeline

    validate_dirs()

    # Resolve collection: explicit --collection → data-dir convention → default.
    if args.collection:
        collection_name = args.collection
    elif args.data_dir:
        collection_name = _derive_collection_from_data_dir(args.data_dir)
    else:
        # No --data-dir given either: use the configured default collection.
        collection_name = DEFAULT_COLLECTION

    print(f"Query: {args.query}")

    # Obtain the two-stage pipeline wrapper (no LLM involved here yet).
    # The wrapper orchestrates ANN retrieval + cross-encoder reranking.
    pipeline = get_retrieval_pipeline(collection_name=collection_name)
    nodes = pipeline.retrieve(args.query)

    print("\n--- Retrieved chunks ---")
    for i, node in enumerate(nodes, 1):
        # node.score: relevance score assigned by the reranker (higher = better)
        print(f"\n{i}. Score: {node.score:.3f}")
        print(f"   ChunkId: {_chunk_id(node)}")
        print(f"   Source: {node.metadata.get('source', 'unknown')}")
        print(f"   Text: {node.text[:200]}...")

    if args.no_generate:
        return

    # Generation step: ask the configured LLM backend for an answer
    # grounded in the retrieved chunks.
    from .generate import generate_answer
    from .llm_client import LLMClient

    llm = LLMClient.from_config()
    try:
        answer = generate_answer(args.query, nodes, llm)
    finally:
        llm.close()

    print("\n--- Answer ---")
    print(answer)


def _build_parser():
    # Import defaults lazily to keep `--help` fast in case build_corpus pulls
    # in heavier dependencies in the future.
    from .build_corpus import DEFAULT_CHUNKS_PER_DOC, DEFAULT_CORPUS_DIR

    parser = argparse.ArgumentParser(
        prog="neorag",
        description="Simple RAG CLI with high-accuracy retrieval.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Create required directories (index, etc.) and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # index
    p_index = subparsers.add_parser(
        "index",
        help="Build vector index from chunks.",
        description="Build vector index from chunks.",
    )
    p_index.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing markdown chunks (determines collection name via convention).",
    )
    p_index.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name. If omitted, derived from --data-dir name.",
    )
    p_index.set_defaults(func=_cmd_index)

    # build-corpus
    p_bc = subparsers.add_parser(
        "build-corpus",
        help="Build the synthetic parent-document corpus over DiscoResearch/germanrag.",
        description=(
            "Build the synthetic parent-document corpus over DiscoResearch/germanrag. "
            "See neorag.build_corpus for the provenance rationale. Produces "
            "doc_{k:05d}.md files and a provenance.jsonl sidecar under corpus-dir. "
            "The target directory is wiped on every invocation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_bc.add_argument(
        "--data-dir",
        required=True,
        help="Target directory for synthetic parent documents (wiped on each run).",
    )
    p_bc.add_argument(
        "--chunks-per-doc",
        default=DEFAULT_CHUNKS_PER_DOC,
        type=int,
        help="Number of consecutive germanrag chunks grouped into one parent doc.",
    )
    p_bc.add_argument(
        "--limit-chunks",
        default=None,
        type=int,
        help="Stop after this many unique chunks (default: no limit).",
    )
    p_bc.add_argument(
        "--limit-docs",
        default=None,
        type=int,
        help="Stop after this many parent documents (default: no limit).",
    )
    p_bc.set_defaults(func=_cmd_build_corpus)

    # query
    p_query = subparsers.add_parser(
        "query",
        help="Query the RAG system.",
        description=(
            "Query the RAG system. Runs a two-stage retrieval pipeline: "
            "1. ANN search in Qdrant to get initial candidate chunks. "
            "2. Cross-encoder reranking to select the most relevant results."
        ),
    )
    p_query.add_argument("query", help="Query string")
    p_query.add_argument(
        "--data-dir",
        default=None,
        help="Corpus directory whose collection to query. "
        "If omitted, uses DEFAULT_COLLECTION from config.",
    )
    p_query.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name. If omitted, derived from --data-dir or config default.",
    )
    p_query.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip the LLM generation step; only print retrieved chunks.",
    )
    p_query.set_defaults(func=_cmd_query)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.bootstrap:
        from .auxiliary import ensure_dirs

        ensure_dirs()
        print("Bootstrap complete: created required directories.")
        return

    if not getattr(args, "func", None):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
