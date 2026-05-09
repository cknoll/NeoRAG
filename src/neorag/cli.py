import click
from pathlib import Path
from .loader import load_chunks
from .indexer import build_index
from .retriever import get_retrieval_pipeline
from .build_corpus import (
    build_corpus,
    DEFAULT_CHUNKS_PER_DOC,
    DEFAULT_CORPUS_DIR,
)

@click.group()
@click.option('--bootstrap', is_flag=True, help='Create required directories (index, etc.) and exit.')
def cli(bootstrap):
    """Simple RAG CLI with high-accuracy retrieval."""
    if bootstrap:
        from .config import ensure_dirs
        ensure_dirs()
        click.echo("Bootstrap complete: created required directories.")
        return

@cli.command()
@click.option('--data-dir', default='data', help='Directory containing markdown chunks')
def index(data_dir):
    """Build vector index from chunks."""
    from .config import validate_dirs
    validate_dirs()
    click.echo(f"Loading chunks from {data_dir}...")
    documents = load_chunks(Path(data_dir))
    click.echo(f"Loaded {len(documents)} documents")

    click.echo("Building index...")
    build_index(documents)
    click.echo("Index built successfully!")

@cli.command("build-corpus")
@click.option(
    "--corpus-dir",
    default=str(DEFAULT_CORPUS_DIR),
    show_default=True,
    help="Target directory for synthetic parent documents (wiped on each run).",
)
@click.option(
    "--chunks-per-doc",
    default=DEFAULT_CHUNKS_PER_DOC,
    show_default=True,
    type=int,
    help="Number of consecutive germanrag chunks grouped into one parent doc.",
)
@click.option(
    "--limit-chunks",
    default=None,
    type=int,
    help="Stop after this many unique chunks (default: no limit).",
)
@click.option(
    "--limit-docs",
    default=None,
    type=int,
    help="Stop after this many parent documents (default: no limit).",
)
def build_corpus_cmd(corpus_dir, chunks_per_doc, limit_chunks, limit_docs):
    """Build the synthetic parent-document corpus over DiscoResearch/germanrag.

    See ``neorag.build_corpus`` for the provenance rationale. Produces
    ``doc_{k:05d}.md`` files and a ``provenance.jsonl`` sidecar under
    ``corpus-dir``. The target directory is wiped on every invocation.
    """
    click.echo(f"Building synthetic corpus in {corpus_dir} ...")
    summary = build_corpus(
        corpus_dir=Path(corpus_dir),
        chunks_per_doc=chunks_per_doc,
        limit_chunks=limit_chunks,
        limit_docs=limit_docs,
    )
    click.echo(
        f"Wrote {summary['n_docs']} parent documents "
        f"({summary['n_chunks']} chunks) to {summary['corpus_dir']}"
    )


@cli.command()
@click.argument('query')
def query(query):
    """Query the RAG system.

    Runs a two-stage retrieval pipeline:
    1. ANN search in Qdrant to get initial candidate chunks.
    2. Cross-encoder reranking to select the most relevant results.
    """
    from .config import validate_dirs
    validate_dirs()
    click.echo(f"Query: {query}")

    # Obtain the two-stage pipeline wrapper (no LLM involved). The wrapper
    # orchestrates ANN retrieval + cross-encoder reranking internally.
    pipeline = get_retrieval_pipeline()
    nodes = pipeline.retrieve(query)

    click.echo("\n--- Results ---")
    for i, node in enumerate(nodes, 1):
        # node.score: relevance score assigned by the reranker (higher = better)
        click.echo(f"\n{i}. Score: {node.score:.3f}")
        click.echo(f"   Source: {node.metadata.get('source', 'unknown')}")
        click.echo(f"   Text: {node.text[:200]}...")

def main():
    cli()

if __name__ == '__main__':
    cli()
