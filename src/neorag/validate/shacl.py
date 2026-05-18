"""SHACL validator.

Converts a parsed :class:`Answer` into an in-memory RDF graph and
validates it against ``shapes/answer.ttl`` plus a dynamically injected
``sh:in`` constraint that restricts citation keys to the set of chunks
actually retrieved for the current query.

The "cited chunk must be retrieved" constraint deliberately overlaps
with :mod:`neorag.validate.groundedness`. The redundancy is *not* a
bug: it serves a demonstration purpose for FF2, showing that the same
constraint can be enforced both procedurally (Python) and declaratively
(SHACL over an RDF view of the answer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Set, Tuple

from .schema import Answer
from .violation import Violation


_SHAPES_PATH = Path(__file__).parent / "shapes" / "answer.ttl"

NRG_NS = "https://neorag.example/ns#"


def _citation_key(doc_id: str, chunk_idx_in_doc: int) -> str:
    """Encode a citation as a single string usable in `sh:in`."""
    return f"{doc_id}#{chunk_idx_in_doc}"


def _retrieved_keys(retrieved_nodes: Iterable[Any]) -> Set[str]:
    """Extract citation keys from retrieved nodes (mirrors groundedness)."""
    keys: Set[str] = set()
    for node in retrieved_nodes:
        md = getattr(node, "metadata", None) or {}
        doc_id = md.get("doc_id") or md.get("source")
        if isinstance(doc_id, str) and doc_id.endswith(".md"):
            doc_id = doc_id[: -len(".md")]
        chunk_idx = md.get("chunk_idx_in_doc")
        if chunk_idx is None:
            chunk_idx = md.get("chunk_idx")
        if doc_id is None or chunk_idx is None:
            continue
        try:
            keys.add(_citation_key(str(doc_id), int(chunk_idx)))
        except (TypeError, ValueError):
            continue
    return keys


def _answer_to_rdf(answer: Answer):
    """Convert ``answer`` into an in-memory rdflib graph."""
    from rdflib import Graph, Literal, Namespace, RDF, URIRef
    from rdflib.namespace import XSD

    nrg = Namespace(NRG_NS)
    g = Graph()
    g.bind("nrg", nrg)

    for ci, claim in enumerate(answer.claims):
        for cj, cit in enumerate(claim.citations):
            cit_uri = URIRef(f"{NRG_NS}citation/{ci}/{cj}")
            g.add((cit_uri, RDF.type, nrg.Citation))
            g.add((cit_uri, nrg.docId, Literal(cit.doc_id, datatype=XSD.string)))
            g.add(
                (
                    cit_uri,
                    nrg.chunkIdxInDoc,
                    Literal(cit.chunk_idx_in_doc, datatype=XSD.integer),
                )
            )
            g.add(
                (
                    cit_uri,
                    nrg.citationKey,
                    Literal(
                        _citation_key(cit.doc_id, cit.chunk_idx_in_doc),
                        datatype=XSD.string,
                    ),
                )
            )
    return g


def _build_shapes_graph(retrieved_keys: Set[str]):
    """Load static shapes and inject a dynamic `sh:in` over citationKey."""
    from rdflib import BNode, Graph, Literal, Namespace, RDF, URIRef
    from rdflib.collection import Collection
    from rdflib.namespace import XSD

    sh = Namespace("http://www.w3.org/ns/shacl#")
    nrg = Namespace(NRG_NS)

    shapes = Graph()
    shapes.parse(_SHAPES_PATH, format="turtle")
    shapes.bind("sh", sh)
    shapes.bind("nrg", nrg)

    # Build an `sh:in` list of allowed citation keys and attach it as an
    # extra property shape on nrg:CitationShape. If retrieval returned
    # nothing we still need a non-empty list (rdflib refuses empty
    # rdf:List heads); use a sentinel that no real key will match,
    # which guarantees that any citation triggers a violation.
    keys = sorted(retrieved_keys) if retrieved_keys else ["__no_retrieved_chunks__"]

    list_head = BNode()
    Collection(
        shapes,
        list_head,
        [Literal(k, datatype=XSD.string) for k in keys],
    )

    prop_shape = BNode()
    shapes.add((prop_shape, sh.path, nrg.citationKey))
    shapes.add((prop_shape, getattr(sh, "in"), list_head))

    citation_shape = URIRef(f"{NRG_NS}CitationShape")
    shapes.add((citation_shape, sh.property, prop_shape))

    return shapes


def validate_shacl(
    answer: Answer,
    retrieved_nodes: Iterable[Any],
) -> List[Violation]:
    """Validate ``answer`` against the SHACL shapes; return violations.

    Returns ``Violation(kind="shacl", ...)`` entries. An empty list
    means the answer's RDF view conforms to all shapes.
    """
    try:
        from pyshacl import validate as pyshacl_validate
    except ImportError as e:  # pragma: no cover
        return [
            Violation(
                kind="shacl",
                message=(
                    "pyshacl is not installed; SHACL validation skipped. "
                    "Install it via requirements.txt."
                ),
            )
        ]

    data_graph = _answer_to_rdf(answer)
    shapes_graph = _build_shapes_graph(_retrieved_keys(retrieved_nodes))

    conforms, _results_graph, results_text = pyshacl_validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        debug=False,
    )

    if conforms:
        return []

    return [
        Violation(
            kind="shacl",
            message=f"SHACL validation failed:\n{results_text}",
        )
    ]
