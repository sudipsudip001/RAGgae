from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
from typing import List, Dict
import pandas as pd
import os

class Ragged:
    def __init__(
        self,
        dataset: List[Dict],
        embedder: str = "thenlper/gte-small",
        llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    ) -> None:
        self.EMBEDDING_MODEL_NAME = embedder
        self.LLM_MODEL_NAME = llm_model
        self._ragas_llm = None
        self._embedder: HuggingFaceEmbeddings | None = None
        # Ensure the dataset has the required columns: question, contexts, answer, ground_truth
        for i, row in enumerate(dataset):
            for key in ["question", "contexts", "answer", "ground_truth"]:
                if key not in row:
                    print(f"Row {i} missing key: {key}")
                elif row[key] is None:
                    print(f"Row {i} has None for: {key}")
            if isinstance(row.get("contexts"), list) and len(row["contexts"]) == 0:
                print(f"Row {i} has empty contexts list")
            if isinstance(row.get("contexts"), str):
                print(f"Row {i}: contexts is a string, should be List[str]")
        self.dataset = Dataset.from_list([
            {
                "user_input":          row["question"],
                "retrieved_contexts":  row["contexts"],
                "response":            row["answer"],
                "reference":           row["ground_truth"],
            }
            for row in dataset
        ])
        self.metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    @property
    def ragas_llm(self):
        # if self._ragas_llm is None:
        #     hf_llm = HuggingFaceEndpoint(
        #         repo_id=self.LLM_MODEL_NAME,
        #         task="text-generation",
        #         temperature=0.1,
        #         max_new_tokens=512,
        #     )
        #     self._ragas_llm = LangchainLLMWrapper(hf_llm)
        if self._ragas_llm is None:
            self._ragas_llm = LangchainLLMWrapper(
                ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    api_key=os.getenv("GROQ_API_KEY")
                )
            )
        return self._ragas_llm
    
    @property
    def embedder(self) -> HuggingFaceEmbeddings:
        if self._embedder is None:
            self._embedder = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL_NAME)
            )
        return self._embedder

    def score(self) -> pd.DataFrame:
        # print("Dataset is: ", self.dataset)
        # print("Metrics is: ", self.metrics)
        # print("The LLM is: ", self.ragas_llm)
        # print("Embedder is: ", self.embedder)
        results = evaluate(
            self.dataset,
            metrics=self.metrics,
            llm=self.ragas_llm,
            embeddings=self.embedder,
        )

        return results.to_pandas()
