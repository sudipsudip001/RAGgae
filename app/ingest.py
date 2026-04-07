from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

class IngestPdf:
    """
        This class is supposed to take in PDF files and return the output in text format.
    """

    def __init__(self, files: List[str]) -> None:
        self.PDF_LIST = files

    def extract_text(self) -> List[Document]:
        RAW_KNOWLEDGE_BASE = []
        for pdf_path in self.PDF_LIST:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            RAW_KNOWLEDGE_BASE.extend(pages)
        return RAW_KNOWLEDGE_BASE
