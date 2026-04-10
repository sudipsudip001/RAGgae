from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Optional
import torch

class Ingestor:
    MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n", "\n\n", "\n", " ", ""]

    def __init__(
        self,
        files: List[str],
        chunk_size: int,
        chunk_overlap: int,
        embedder_model: str = "thenlper/gte-small",
    ) -> None:
        self.PDF_LIST = files
        self.CHUNK_SIZE = chunk_size
        self.CHUNK_OVERLAP = chunk_overlap
        self.EMBEDDING_MODEL_NAME = embedder_model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._raw_knowledge_base: Optional[List[Document]] = None
        self._vector_database: Optional[FAISS] = None

    @property
    def raw_knowledge_base(self) -> List[Document]:
        if self._raw_knowledge_base is None:
            self._raw_knowledge_base = []
            for pdf_path in self.PDF_LIST:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                for page in pages:
                    self._raw_knowledge_base.append(
                        Document(
                            page_content=page.page_content,
                            metadata={
                                "source": pdf_path,
                                "page": page.metadata["page"]+1
                            }
                        )
                    )
        return self._raw_knowledge_base

    @property
    def vector_database(self) -> FAISS:
        if self._vector_database is None:
            chunkz = self.chunker()
            embedder = HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL_NAME,
                multi_process=False,
                model_kwargs={"device": self._device},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._vector_database = FAISS.from_documents(
                chunkz,
                embedder,
                distance_strategy=DistanceStrategy.COSINE,
            )
        return self._vector_database

    def chunker(self) -> List[Document]:
        docs = self.raw_knowledge_base
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL_NAME),
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            add_start_index=True,
            strip_whitespace=True,
            separators=self.MARKDOWN_SEPARATORS,
        )
        seen, chunks = set(), []
        for doc in text_splitter.split_documents(docs):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                chunks.append(doc)
        return chunks
