from typing import List
from langchain_community.vectorstores import FAISS

class Retriever:
    def __init__(
        self,
        vector_db: FAISS,
        embedder_model: str = "thenlper/gte-small",
    ) -> None:
        self.EMBEDDING_MODEL_NAME = embedder_model
        self.VECTOR_DATABASE = vector_db

    def retrieve_docs(self, user_query: str) -> List:
        retrieved_docs = self.VECTOR_DATABASE.similarity_search(query=user_query, k=5)
        return retrieved_docs
