from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
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
        self.dataset = Dataset.from_list(dataset)

    @property
    def ragas_llm(self):
        if self._ragas_llm is None:
            hf_llm = HuggingFaceEndpoint(
                repo_id=self.LLM_MODEL_NAME,
                task="text-generation",
                temperature=0.1,
                max_new_tokens=512,
            )
            self._ragas_llm = LangchainLLMWrapper(hf_llm)
        return self._ragas_llm
    
    @property
    def embedder(self) -> HuggingFaceEmbeddings:
        if self._embedder is None:
            # Use the instance variable instead of a hardcoded string
            self._embedder = HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL_NAME 
            )
        return self._embedder

    def score(self) -> pd.DataFrame:
        results = evaluate(
            self.dataset,
            metrics=[
                faithfulness,      # No parentheses
                answer_relevancy,   # No parentheses
                context_precision, # No parentheses
                context_recall,    # No parentheses
            ],
            llm=self.ragas_llm,
            embeddings=self.embedder,
        )

        return results.to_pandas()