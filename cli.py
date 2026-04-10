import click
from pathlib import Path
from rag.loader import load_chunks
from rag.indexer import build_index
from rag.retriever import get_query_engine

@click.group()
def cli():
    """Simple RAG CLI with high-accuracy retrieval."""
    pass

@cli.command()
@click.option('--data-dir', default='data', help='Directory containing markdown chunks')
def index(data_dir):
    """Build vector index from chunks."""
    click.echo(f"Loading chunks from {data_dir}...")
    documents = load_chunks(Path(data_dir))
    click.echo(f"Loaded {len(documents)} documents")

    click.echo("Building index...")
    build_index(documents)
    click.echo("Index built successfully!")

@cli.command()
@click.argument('query')
def query(query):
    """Query the RAG system."""
    click.echo(f"Query: {query}")
    query_engine = get_query_engine()


    response = query_engine.query(query)

    click.echo("\n--- Results ---")
    for i, node in enumerate(response.source_nodes, 1):
        click.echo(f"\n{i}. Score: {node.score:.3f}")
        click.echo(f"   Source: {node.metadata.get('source', 'unknown')}")
        click.echo(f"   Text: {node.text[:200]}...")

if __name__ == '__main__':
    cli()
