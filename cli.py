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


    # TODO-AIDER: the following statement causes an error:
    response = query_engine.query(query)

    # The traceback of this error is listed below. Please fix the error. Traceback (most recent call last):
    '''
File ~/Nextcloud/tmp2023/2025-semi-local-rag/neo-rag/cli.py:31, in query(query)
     28 click.echo(f"Query: {query}")
     29 query_engine = get_query_engine()
---> 31 response = query_engine.query(query)
     33 click.echo("\n--- Results ---")
     34 for i, node in enumerate(response.source_nodes, 1):

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/instrumentation/dispatcher.py:260, in Dispatcher.span.<locals>.wrapper(func, instance, args, kwargs)
    252 self.span_enter(
    253     id_=id_,
    254     bound_args=bound_args,
   (...)    257     tags=tags,
    258 )
    259 try:
--> 260     result = func(*args, **kwargs)
    261 except BaseException as e:
    262     self.event(SpanDropEvent(span_id=id_, err_str=str(e)))

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/base/base_query_engine.py:52, in BaseQueryEngine.query(self, str_or_query_bundle)
     50     if isinstance(str_or_query_bundle, str):
     51         str_or_query_bundle = QueryBundle(str_or_query_bundle)
---> 52     query_result = self._query(str_or_query_bundle)
     53 dispatcher.event(
     54     QueryEndEvent(query=str_or_query_bundle, response=query_result)
     55 )
     56 return query_result

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/instrumentation/dispatcher.py:260, in Dispatcher.span.<locals>.wrapper(func, instance, args, kwargs)
    252 self.span_enter(
    253     id_=id_,
    254     bound_args=bound_args,
   (...)    257     tags=tags,
    258 )
    259 try:
--> 260     result = func(*args, **kwargs)
    261 except BaseException as e:
    262     self.event(SpanDropEvent(span_id=id_, err_str=str(e)))

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/query_engine/retriever_query_engine.py:189, in RetrieverQueryEngine._query(self, query_bundle)
    185 """Answer a query."""
    186 with self.callback_manager.event(
    187     CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
    188 ) as query_event:
--> 189     nodes = self.retrieve(query_bundle)
    190     response = self._response_synthesizer.synthesize(
    191         query=query_bundle,
    192         nodes=nodes,
    193     )
    194     query_event.on_end(payload={EventPayload.RESPONSE: response})

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/query_engine/retriever_query_engine.py:144, in RetrieverQueryEngine.retrieve(self, query_bundle)
    143 def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
--> 144     nodes = self._retriever.retrieve(query_bundle)
    145     return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/instrumentation/dispatcher.py:260, in Dispatcher.span.<locals>.wrapper(func, instance, args, kwargs)
    252 self.span_enter(
    253     id_=id_,
    254     bound_args=bound_args,
   (...)    257     tags=tags,
    258 )
    259 try:
--> 260     result = func(*args, **kwargs)
    261 except BaseException as e:
    262     self.event(SpanDropEvent(span_id=id_, err_str=str(e)))

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/base/base_retriever.py:243, in BaseRetriever.retrieve(self, str_or_query_bundle)
    238 with self.callback_manager.as_trace("query"):
    239     with self.callback_manager.event(
    240         CBEventType.RETRIEVE,
    241         payload={EventPayload.QUERY_STR: query_bundle.query_str},
    242     ) as retrieve_event:
--> 243         nodes = self._retrieve(query_bundle)
    244         nodes = self._handle_recursive_retrieval(query_bundle, nodes)
    245         retrieve_event.on_end(
    246             payload={EventPayload.NODES: nodes},
    247         )

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/instrumentation/dispatcher.py:260, in Dispatcher.span.<locals>.wrapper(func, instance, args, kwargs)
    252 self.span_enter(
    253     id_=id_,
    254     bound_args=bound_args,
   (...)    257     tags=tags,
    258 )
    259 try:
--> 260     result = func(*args, **kwargs)
    261 except BaseException as e:
    262     self.event(SpanDropEvent(span_id=id_, err_str=str(e)))

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/indices/vector_store/retrievers/retriever.py:101, in VectorIndexRetriever._retrieve(self, query_bundle)
     95     if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
     96         query_bundle.embedding = (
     97             self._embed_model.get_agg_embedding_from_queries(
     98                 query_bundle.embedding_strs
     99             )
    100         )
--> 101 return self._get_nodes_with_embeddings(query_bundle)

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/core/indices/vector_store/retrievers/retriever.py:177, in VectorIndexRetriever._get_nodes_with_embeddings(self, query_bundle_with_embeddings)
    173 def _get_nodes_with_embeddings(
    174     self, query_bundle_with_embeddings: QueryBundle
    175 ) -> List[NodeWithScore]:
    176     query = self._build_vector_store_query(query_bundle_with_embeddings)
--> 177     query_result = self._vector_store.query(query, **self._kwargs)
    178     return self._build_node_list_from_query_result(query_result)

File /media/data2/venvs/ragenv/lib/python3.12/site-packages/llama_index/vector_stores/qdrant/base.py:836, in QdrantVectorStore.query(self, query, **kwargs)
    834     return self.parse_to_query_result(response[0])
    835 else:
--> 836     response = self._client.search(
    837         collection_name=self.collection_name,
    838         query_vector=query_embedding,
    839         limit=query.similarity_top_k,
    840         query_filter=query_filter,
    841     )
    842     return self.parse_to_query_result(response)

AttributeError: 'QdrantClient' object has no attribute 'search'
    '''

    click.echo("\n--- Results ---")
    for i, node in enumerate(response.source_nodes, 1):
        click.echo(f"\n{i}. Score: {node.score:.3f}")
        click.echo(f"   Source: {node.metadata.get('source', 'unknown')}")
        click.echo(f"   Text: {node.text[:200]}...")

if __name__ == '__main__':
    cli()
