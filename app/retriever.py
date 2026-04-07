from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
from typing import List, Optional
from ingest import IngestPdf
import os
from dotenv import load_dotenv

class Retriever:
    def __init__(self) -> None:
        self._raw_knowledge_base = None
        self.MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n", "\n\n", "\n", " ", ""]
        self.EMBEDDING_MODEL_NAME = "thenlper/gte-small"
        load_dotenv()

    @property
    def raw_knowledge_base(self) -> List[Document]:
        if self._raw_knowledge_base is None:
            ingestor = IngestPdf(["../docs/crime_act.pdf", "../docs/interim_government_act.pdf", "../docs/labor_act.pdf"])
            self._raw_knowledge_base = ingestor.extract_text()
        return self._raw_knowledge_base


    def split_documents(
        self,
        chunk_size: int,
        knowledge_base: List[Document],
        tokenizer_name: Optional[str] = None,
    ) -> List[Document]:

        model_to_use = tokenizer_name or self.EMBEDDING_MODEL_NAME
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(model_to_use),
            chunk_size=chunk_size,
            chunk_overlap=50,
            add_start_index=True,
            strip_whitespace=True,
            separators=self.MARKDOWN_SEPARATORS,
        )

        docs_processed = text_splitter.split_documents(knowledge_base)

        unique_texts = set()
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts.add(doc.page_content)
                docs_processed_unique.append(doc)

        return docs_processed_unique

    def build_vector_database(self, tokenizer_name: Optional[str] = None) -> FAISS:
        model_to_use = tokenizer_name or self.EMBEDDING_MODEL_NAME

        docs_processed = self.split_documents(
            512,
            self.raw_knowledge_base,
            tokenizer_name=model_to_use,
        )
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_to_use,
            multi_process=False,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vector_db = FAISS.from_documents(
            docs_processed,
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )
        return vector_db

    def retrieve_docs(self, vector_db, user_query: str) -> List:
        retrieved_docs = vector_db.similarity_search(query=user_query, k=5)
        return retrieved_docs
