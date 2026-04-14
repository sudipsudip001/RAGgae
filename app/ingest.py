from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Optional
import torch
import numpy as np

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

    def semantic_chunker(
        self,
        knowledge_base: List[Document],
    ) -> List[Document]:
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL_NAME
        )
        text_splitter = SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="percentile", # where to split
            breakpoint_threshold_amount=95 # sensitivity of the split
        )
        docs_processed = text_splitter.split_documents(knowledge_base)
        return docs_processed


    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def embedding_chunk_documents(self, docs, embedder, threshold=0.75):
        final_chunks = []

        for doc in docs:
            # Better splitting: captures ! and ? as well
            import re
            sentences = re.split(r'(?<=[.!?]) +', doc.page_content)
            if not sentences: continue

            embeddings = embedder.embed_documents(sentences)

            current_chunk_sentences = [sentences[0]]
            current_chunk_embedding = embeddings[0]

            for i in range(1, len(sentences)):
                # Compare current sentence to the previous one
                sim = self.cosine_similarity(embeddings[i-1], embeddings[i])

                if sim > threshold:
                    current_chunk_sentences.append(sentences[i])
                else:
                    # Close off the current chunk
                    final_chunks.append(
                        Document(
                            page_content=" ".join(current_chunk_sentences),
                            metadata=doc.metadata
                        )
                    )
                    current_chunk_sentences = [sentences[i]]

            # Catch the last remaining chunk
            if current_chunk_sentences:
                final_chunks.append(
                    Document(page_content=" ".join(current_chunk_sentences), metadata=doc.metadata)
                )

        return final_chunks
