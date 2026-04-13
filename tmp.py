

import datasets



dataset = datasets.load_dataset("DiscoResearch/germanrag", split="train")


from ragas import evaluate
from ragas.metrics.collections import (
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


from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy


# In[13]:


import tomllib
with open("config.toml", "rb") as fp:
    config = tomllib.load(fp)


# In[18]:


from ragas.llms import llm_factory
from openai import OpenAI

client = OpenAI(
    api_key=config["openrouter_api_key"],
    base_url="https://openrouter.ai/api/v1",
)

llm = llm_factory("gemini-flash-1.5", client=client)

# now use this llm

metrics = [ContextPrecision(llm=llm), ]#, ContextRecall(), Faithfulness(), AnswerRelevancy()]

# this causes an TypeError:
result = evaluate(
    ragas_dataset,
    metrics=metrics,
)
