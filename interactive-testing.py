"""
This file is meant to be run manually in interactive mode in vs code.
"""
# %%


import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Importing .* from 'ragas\.metrics' is deprecated",
    category=DeprecationWarning,
)


#! %load_ext ipydex.displaytools
displaytools_extension = "works"  ##:

# Current Ragas versions have an API mismatch:
# The `evaluate()` function expects old-style metrics from `ragas.metrics`
# (which are subclasses of `ragas.metrics.base.Metric`).
# However, the new `ragas.metrics.collections` metrics are NOT subclasses of `ragas.metrics.base.Metric`.
# To make this code future-compatible, when `ragas.metrics` imports are finally removed in v1.0,
# you will need to:
# 1. Update imports to `from ragas.metrics.collections import ...`
# 2. Rework the evaluation loop to call `metric.single_turn_score()` or `metric.score()` directly on each sample,
#    or ensure `evaluate()` is updated in Ragas to properly handle the new metric types.
#    Alternatively, if a new `evaluate_v2()` function exists, use that.

import datasets


# %%


dataset = datasets.load_dataset("DiscoResearch/germanrag", split="train")

# %%

from ragas import evaluate
from ragas.metrics import (
    context_recall,
    faithfulness,
    answer_relevancy,
    NonLLMContextPrecisionWithReference,
)
import pandas as pd


from llama_index.core import Document, QueryBundle
from rag.indexer import build_index
from rag.retriever import get_query_engine

GERMANRAG_COLLECTION = "germanrag_chunks"

# --- Build a Qdrant index from the GermanRAG contexts (once) ---
# Extract all unique context passages and index them into a separate collection
# so the retriever searches GermanRAG content, not the default podcast data.
_all_contexts = set()
for _row in dataset:
    for _ctx in _row["contexts"]:
        _all_contexts.add(_ctx)

_germanrag_docs = [
    Document(text=ctx, metadata={"source": "germanrag"})
    for ctx in _all_contexts
]
import os

FLAG_FILE = "index/.germanrag_indexed"

if not os.path.exists(FLAG_FILE):
    print(f"Indexing {len(_germanrag_docs)} unique GermanRAG contexts into '{GERMANRAG_COLLECTION}' ...")
    #build_index(_germanrag_docs, collection_name=GERMANRAG_COLLECTION)
    print("Done.")
    with open(FLAG_FILE, "w") as f:
        f.write("indexed")
else:
    print(f"GermanRAG index already exists. Skipping indexing.")

# %%

# Initialise the two-stage retrieval pipeline against the GermanRAG collection
_base_retriever, _reranker = get_query_engine(collection_name=GERMANRAG_COLLECTION)

def run_my_rag_system(question, llm=None):
    """Run the actual RAG pipeline: retrieve, rerank, then generate an answer.

    Parameters
    ----------
    question : str
        The user query.
    llm : langchain LLM, optional
        If provided, used to generate an answer from the retrieved context.
        Otherwise a placeholder answer is returned.
    """
    query_bundle = QueryBundle(question)

    # Stage 1: ANN retrieval from Qdrant
    nodes = _base_retriever.retrieve(query_bundle)
    # Stage 2: cross-encoder reranking
    nodes = _reranker.postprocess_nodes(nodes, query_bundle)

    retrieved_docs = [node.text for node in nodes]

    if llm is not None:
        context = "\n\n".join(retrieved_docs)
        prompt = (
            f"Beantworte die folgende Frage basierend auf dem gegebenen Kontext.\n\n"
            f"Kontext:\n{context}\n\nFrage: {question}\nAntwort:"
        )
        generated_answer = llm.invoke(prompt).content
    else:
        generated_answer = ""

    return generated_answer, retrieved_docs

import tomllib
with open("config.toml", "rb") as fp:
    config = tomllib.load(fp)

# %%

from langchain_openai import ChatOpenAI

langchain_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    api_key=config["openrouter_api_key"],
    base_url="https://openrouter.ai/api/v1",
)

# 2. Testdaten sammeln (z.B. für die ersten 10 Einträge)
N = 10
test_results = []
for i in range(N):
    row = dataset[i]
    q = row['question']
    gt = row['answer'] # Das ist die Gold-Antwort aus GermanRAG
    ref_contexts = row['contexts']  # Reference contexts from GermanRAG

    # Dein System fragen
    pred_answer, pred_contexts = run_my_rag_system(q, llm=langchain_llm)

    _result = {
        "question": q,
        "answer": pred_answer,
        "contexts": pred_contexts,
        "reference_contexts": ref_contexts,
        "ground_truth": gt
    }

    test_results.append(_result)

# %%

# In ein Hugging Face Dataset konvertieren
from datasets import Dataset
ragas_input_df = pd.DataFrame(test_results)
ragas_dataset = Dataset.from_pandas(ragas_input_df)

metrics = [NonLLMContextPrecisionWithReference()]

# %%

# only run if needed
if 1:
    result = evaluate(
        ragas_dataset,
        metrics=metrics,
    )

# %%


print(123)
print(4)

# %%
