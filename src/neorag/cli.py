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
    """Query the RAG system: retrieve → generate → validate → (refine)*.

    Runs a two-stage retrieval pipeline (ANN + cross-encoder reranking),
    then generates a structured answer with citations, validates it, and
    optionally refines it in a loop until all violations are resolved or
    --max-iter is reached.

    Pass --no-generate to skip generation entirely (retrieval-only debug).
    Pass --no-refine to validate but skip the refinement loop.
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
        collection_name = DEFAULT_COLLECTION

    print(f"Query: {args.query}")

    pipeline = get_retrieval_pipeline(collection_name=collection_name)
    nodes = pipeline.retrieve(args.query)

    print("\n--- Retrieved chunks ---")
    for i, node in enumerate(nodes, 1):
        print(f"\n{i}. Score: {node.score:.3f}")
        print(f"   ChunkId: {_chunk_id(node)}")
        print(f"   Source: {node.metadata.get('source', 'unknown')}")
        print(f"   Text: {node.text[:200]}...")

    if args.no_generate:
        return

    from .generate import generate_structured_answer
    from .llm_client import LLMClient
    from .validate import validate
    from .validate.refine import refine

    llm = LLMClient.from_config()
    try:
        gen_result = generate_structured_answer(args.query, nodes, llm)

        if gen_result.parsed is None:
            print(f"\n--- Generation failed ---")
            print(f"Could not parse LLM response: {gen_result.parse_error}")
            return

        violations = validate(gen_result.raw_text, nodes)
        final_answer = gen_result.parsed
        history = []

        if violations and not args.no_refine:
            final_answer, history = refine(
                query=args.query,
                retrieved_nodes=nodes,
                answer=gen_result.parsed,
                violations=violations,
                llm=llm,
                max_iter=args.max_iter,
                feedback_granularity=args.feedback_granularity,
            )
    finally:
        llm.close()

    # Remaining violations: from the last history entry that parsed, or the
    # original violations when no refinement ran or every retry failed.
    if history:
        last_valid = next((h for h in reversed(history) if h.answer is not None), None)
        remaining = last_valid.violations_out if last_valid else violations
    else:
        remaining = violations

    print("\n--- Answer ---")
    print(final_answer.summary)
    print("\nClaims:")
    for idx, claim in enumerate(final_answer.claims, 1):
        cites = ", ".join(f"[{c.doc_id}#{c.chunk_idx_in_doc}]" for c in claim.citations)
        print(f"  {idx}. {claim.text}  ({cites})")

    print("\n--- Validation ---")
    if history:
        print(f"Refinement iterations: {len(history)}")
    if not remaining:
        print("No violations.")
    else:
        print(f"{len(remaining)} violation(s):")
        for v in remaining:
            loc = f" {v.location}:" if v.location else ""
            print(f"  [{v.kind}]{loc} {v.message}")


def _cmd_eval(args):
    """Run the full evaluation pipeline on a labelled test set.

    ``test_path`` may be either a JSON test file (existing behaviour) or a
    TOML settings file (e.g. ``configs/demo.toml``).  When a TOML file is
    given, evaluation settings and the LLM backend are taken from the file;
    CLI flags override individual settings values.
    """
    from pathlib import Path as _Path
    from .evaluate import evaluate

    path = _Path(args.test_path)

    if path.suffix.lower() == ".toml":
        from .config import load_settings
        from .llm_client import StubBackend

        settings = load_settings(path)

        # CLI flags override TOML settings when explicitly provided.
        test_path = _Path(settings.eval.test_path) if settings.eval.test_path else None
        if test_path is None:
            print("Error: TOML config does not specify [eval] test_path.")
            return
        max_iter = args.max_iter if args.max_iter != 3 else settings.eval.max_iter
        feedback_granularity = (
            args.feedback_granularity
            if args.feedback_granularity != "per_violation"
            else settings.eval.feedback_granularity
        )
        no_refine = args.no_refine or settings.eval.no_refine
        runs_dir = _Path(settings.eval.runs_dir)

        # Wire up the LLM backend from settings.
        if settings.llm.provider == "stub":
            llm_backend = StubBackend(
                canned_response='{"summary": "Stub answer.", "claims": []}',
                model=settings.llm.model,
            )
        else:
            llm_backend = None  # evaluate() creates LLMClient.from_config()
    else:
        test_path = path
        max_iter = args.max_iter
        feedback_granularity = args.feedback_granularity
        no_refine = args.no_refine
        runs_dir = _Path(args.runs_dir)
        llm_backend = None

    run_dir = evaluate(
        test_path=test_path,
        runs_dir=runs_dir,
        max_iter=max_iter,
        feedback_granularity=feedback_granularity,
        no_refine=no_refine,
        _llm_backend=llm_backend,
    )
    return run_dir


def _build_parser():
    # Import defaults lazily to keep `--help` fast in case build_corpus pulls
    # in heavier dependencies in the future.
    from .build_corpus import DEFAULT_CHUNKS_PER_DOC

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
            "Query the RAG system. Runs: retrieval → structured generation → "
            "validation → optional self-refinement loop."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Skip LLM generation entirely; only print retrieved chunks.",
    )
    p_query.add_argument(
        "--no-refine",
        action="store_true",
        help="Validate the answer but skip the self-refinement loop.",
    )
    p_query.add_argument(
        "--max-iter",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of refinement iterations.",
    )
    p_query.add_argument(
        "--feedback-granularity",
        choices=["coarse", "per_violation"],
        default="per_violation",
        help="How violations are described in the refinement prompt.",
    )
    p_query.set_defaults(func=_cmd_query)

    # eval
    p_eval = subparsers.add_parser(
        "eval",
        help="Evaluate retrieval + generation quality on a labelled test set.",
        description=(
            "Evaluate NeoRAG on a JSON test file or a TOML settings file. "
            "Measures MRR@10, Recall@5, and constraint-violation rates "
            "before/after refinement. Dumps a timestamped run directory. "
            "Example: neorag eval configs/demo.toml"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_eval.add_argument(
        "test_path",
        help=(
            "Path to a JSON test file ({query, expected_sources} list) "
            "or a TOML settings file (see configs/demo.toml)."
        ),
    )
    p_eval.add_argument(
        "--runs-dir",
        default="runs",
        metavar="DIR",
        help="Parent directory for timestamped run folders.",
    )
    p_eval.add_argument(
        "--no-refine",
        action="store_true",
        help="Validate answers but skip the self-refinement loop.",
    )
    p_eval.add_argument(
        "--max-iter",
        type=int,
        default=3,
        metavar="N",
        help="Maximum refinement iterations per query.",
    )
    p_eval.add_argument(
        "--feedback-granularity",
        choices=["coarse", "per_violation"],
        default="per_violation",
        help="How violations are described in the refinement prompt.",
    )
    p_eval.set_defaults(func=_cmd_eval)

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
