# NeoRAG

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()

**NeoRAG** is a small, open-source RAG prototype that demonstrates neurosymbolic answer
validation with a closed self-refinement loop, with a pluggable LLM backend.

---

## Motivation

Large language models hallucinate. Existing mitigation approaches either rely on heavyweight
knowledge graphs (hard to maintain) or leave validation implicit inside the model itself
(opaque, non-auditable). NeoRAG takes a different stance: answers are parsed into a typed
schema, converted to an ephemeral RDF view, and validated against lightweight, human-readable
SHACL constraints and Python-level groundedness checks — all without a persistent knowledge
graph. Violations are fed back to the LLM in a structured self-refinement loop with
configurable feedback granularity. The goal is a system that is **small, inspectable, and
reproducible** rather than one that is performant at scale.

NeoRAG is the technical Vorarbeit (Preliminary Work) for research question FF2 of the
SoFAKTA proposal: *Can lightweight neurosymbolic constraint checking in a self-refinement
loop measurably reduce hallucination rates while remaining compatible with local-first,
sovereignty-preserving deployments?*

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Ingestion & Indexing                                           │
  │  Markdown / text → chunks + provenance.jsonl → vector index    │
  └────────────────────────────┬────────────────────────────────────┘
                               │ retrieved chunks (doc_id, chunk_idx)
  ┌────────────────────────────▼────────────────────────────────────┐
  │  Generation                                                     │
  │  build_structured_prompt → LLM → Answer (JSON / Pydantic)      │
  └────────────────────────────┬────────────────────────────────────┘
                               │ Answer object
  ┌────────────────────────────▼────────────────────────────────────┐
  │  Neurosymbolic Validation                                       │
  │  ① Pydantic structural check                                   │
  │  ② Groundedness check  (citation ∈ retrieved chunks?)          │
  │  ③ SHACL validation    (RDF view of answer vs. shapes/answer.ttl)│
  │  → list[Violation]                                             │
  └────────────────────────────┬────────────────────────────────────┘
                               │ violations (if any)
  ┌────────────────────────────▼────────────────────────────────────┐
  │  Self-Refinement Loop  (refine.py)                             │
  │  violations → structured feedback → LLM → re-validate …       │
  │  terminates on empty violations or max_iter                    │
  └─────────────────────────────────────────────────────────────────┘
```

The LLM backend is accessed through a thin `LLMBackend` protocol; the shipped
implementations are `LLMClient` (OpenRouter / OpenAI / Anthropic) and `StubBackend` (for
tests and CI).

---

## Quickstart

```bash
git clone https://github.com/cknoll/neorag.git
cd neorag
pip install -e "."

# Build a small synthetic corpus (groups rows from DiscoResearch/germanrag):
neorag build-corpus --limit-docs 10

# Index the corpus (requires a running Qdrant instance):
neorag index --data-dir data/sample_corpus

# Ask a question:
export OPENROUTER_API_KEY=sk-or-...   # or configure a local backend
neorag query "Was ist maschinelles Lernen?"

# Run the evaluation pipeline with the offline stub backend (no API key needed):
neorag eval configs/demo.toml
```

A running [Qdrant](https://qdrant.tech/) instance is expected on `localhost:6333` by
default (configurable in `config.toml`). The `configs/demo.toml` wires up `StubBackend`
so `neorag eval` runs fully offline.

---

## What NeoRAG demonstrates

- **Neurosymbolic answer validation with self-refinement** — the technical core of FF2:
  SHACL shapes over an ephemeral RDF answer view, combined with Pydantic structural checks
  and provenance-based groundedness checks, all feeding into a configurable refinement loop.
- **Provenance-linked chunks** as substrate for groundedness checks: every chunk carries
  `doc_id`, `chunk_idx_in_doc`, `byte_start`, `byte_end`, and a `sha256` hash.
- **Pluggable LLM backend** via the `LLMBackend` protocol — compatible with remote APIs
  (OpenRouter, OpenAI, Anthropic) and, architecturally, with local open-weight models; no
  GPU or local model weights are required for the demo.
- **Configurable feedback granularity** (`coarse` vs. `per_violation`) as the experimental
  factor announced for SoFAKTA AP4.3.
- **Reproducible evaluation** via `configs/` + timestamped `runs/` dumps, covering
  constraint-violation rates before and after refinement, iteration counts, token overhead,
  and retrieval metrics (MRR@10, Recall@5).

---

## What NeoRAG is *not* (yet)

NeoRAG is the **seed**, not the deliverable, of SoFAKTA. The following are explicitly out
of scope for v0.1:

- A persistent property graph ("Wissensbibliothek") — SHACL validation runs on an
  ephemeral RDF view of the answer object only (no graph DB required).
- Agentic multi-agent KG construction with Human-in-the-Loop (SoFAKTA FF1 / AP3.3).
- The Ordnungsdirektiven DSL and its compiler (SoFAKTA FF3 / AP5).
- Hybrid retrieval (BM25, graph expansion, rerankers), OCR, HTML, scanned PDFs.
- A model-size sweep or local-model benchmark (planned for v0.2).
- A polished end-user explanation UI (SoFAKTA AP6).
- RAGAS-style faithfulness / answer-relevancy metrics (optional extra, not yet wired).
- Production-grade security, performance, or scalability hardening.

---

## Reproducibility

All runs are driven by a single TOML config file (model, embedding, chunker, retriever,
validator, refinement policy). Results are written to a structured `runs/` directory
containing a config snapshot, raw LLM outputs, violation reports, and an aggregated
`metrics.json`.

```bash
# Offline demo run (StubBackend, no API key):
neorag eval configs/demo.toml

# Results appear in runs/<timestamp>/
#   config.json      — resolved settings snapshot
#   metrics.json     — MRR@10, Recall@5, violation rates, iteration counts
#   query_<n>.json   — per-query detail (retrieved chunks, answer, violations)
```

See `configs/demo.toml` for the minimal configuration that wires up the stub backend.

---

## How to cite

If you use NeoRAG in your research, please cite:

```bibtex
@software{knoll2025neorag,
  author    = {Knoll, Carsten},
  title     = {{NeoRAG}: Neurosymbolic Answer Validation with Self-Refinement
               for Retrieval-Augmented Generation},
  year      = {2025},
  url       = {https://github.com/cknoll/neorag},
  note      = {Research prototype; Vorarbeit for the SoFAKTA proposal (FF2)},
}
```

A Zenodo DOI will be assigned to the `v0.1.0` tagged release.

---

## License and acknowledgements

NeoRAG is released under the **GNU General Public License v3** (GPL-3.0).
See [LICENSE](LICENSE) for the full text.

Developed by Carsten Knoll at the **Professur Grundlagen der Elektrotechnik**,
Technische Universität Dresden.

NeoRAG informs and is cited as Vorarbeit in the **SoFAKTA** research proposal (BMFTR).
