

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Importing .* from 'ragas\.metrics' is deprecated",
    category=DeprecationWarning,
)

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



dataset = datasets.load_dataset("DiscoResearch/germanrag", split="train")


from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
import pandas as pd

# 1. Deine RAG-Pipeline Funktion (Dummy-Beispiel)
def run_my_rag_system(question):
    # Hier rufst du deinen Retriever und dein LLM auf
    retrieved_docs = ["Das ist ein gefundener Textbaustein aus deiner DB."]
    generated_answer = "Das ist die Antwort deines Systems."
    return generated_answer, retrieved_docs

# 2. Testdaten sammeln (z.B. für die ersten 10 Einträge)
N = 10
test_results = []
for i in range(N):
    row = dataset[i]
    q = row['question']
    gt = row['answer'] # Das ist die Gold-Antwort aus GermanRAG

    # Dein System fragen
    pred_answer, pred_contexts = run_my_rag_system(q)

    test_results.append({
        "question": q,
        "answer": pred_answer,
        "contexts": pred_contexts,
        "ground_truth": gt
    })

# In ein Hugging Face Dataset konvertieren
from datasets import Dataset
ragas_input_df = pd.DataFrame(test_results)
ragas_dataset = Dataset.from_pandas(ragas_input_df)


# In[10]:


import tomllib
with open("config.toml", "rb") as fp:
    config = tomllib.load(fp)

from langchain_openai import ChatOpenAI

langchain_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    api_key=config["openrouter_api_key"],
    base_url="https://openrouter.ai/api/v1",
)

metrics = [context_precision]

result = evaluate(
    ragas_dataset,
    metrics=metrics,
    llm=langchain_llm,
)
