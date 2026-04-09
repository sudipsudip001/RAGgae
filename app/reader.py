from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
import torch
from rerankers import Reranker
from transformers import Pipeline
from typing import Optional

class Reader:
    def __init__(
        self,
        reader_model_name: str = "HuggingFaceH4/zephyr-7b-beta",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self.READER_MODEL_NAME = reader_model_name
        self.RERANKER_NAME = reranker_name
        self.prompt_in_chat_format= [
            {
                "role": "system",
                "content": 
                """
                You are a question answering system that MUST rely ONLY on the provided context.

                STRICT RULES:
                1. Answer ONLY using information found in the context.
                2. DO NOT use prior knowledge.
                3. DO NOT infer or guess missing information.
                4. If the answer is not explicitly stated in the context, respond:
                    "I cannot find the answer in the context."
                """
            },
            {
            "role": "user",
            "content": 
            """
                Context:
                {context}

                Question:
                {question}
            """
            }
        ]

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._model = None
        self._tokenizer = None
        self._reader_llm = None
        self._reranker = None
        self._rag_prompt_template = None

    @property
    def RAG_PROMPT_TEMPLATE(self):
        if self._rag_prompt_template is None:
            self._rag_prompt_template = self.tokenizer.apply_chat_template(
                self.prompt_in_chat_format, tokenize=False, add_generation_prompt=True
            )
        return self._rag_prompt_template

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = Reranker(model_name=self.RERANKER_NAME)
        return self._reranker

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.READER_MODEL_NAME, quantization_config=self.bnb_config
            )
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.READER_MODEL_NAME)
        return self._tokenizer

    @property
    def reader_llm(self):
        if self._reader_llm is None:
            self._reader_llm = pipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                task="text-generation",
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=500,
            )
        return self._reader_llm

    def answer_with_rag(
        self,
        question: str,
        llm: Pipeline,
        knowledge_index: FAISS,
        reranker: Optional[Reranker] = None,
        num_retrieved_docs: int = 5,
        num_docs_final: int = 3,
    ):
        print("===> Retrieving documents...")
        initial_docs = knowledge_index.similarity_search(
            query=question, k=num_retrieved_docs
        )
        relevant_docs = initial_docs
        if reranker:
            print("===> Reranking documents...")
            doc_texts = [doc.page_content for doc in relevant_docs]
            rerank_results = reranker.rank(question, doc_texts)
            reranked_docs = []
            for res in rerank_results.results[:num_docs_final]:
                for doc in relevant_docs:
                    if doc.page_content == res.document:
                        reranked_docs.append(doc)
                        break
        else:
            relevant_docs = relevant_docs[:num_docs_final]
        chunks = []
        context = "\nExtracted documents:\n"
        for i, doc in enumerate(relevant_docs, start=1):
            source = doc.metadata["source"]
            page = doc.metadata["page"]
            chunks.append(
                f"[Chunk {i}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{doc.page_content}"
            )
            context = "\n\n---\n\n".join(chunks)
        final_prompt = self.RAG_PROMPT_TEMPLATE.format(question=question, context=context)
        print("=> Generating answer...")
        answer = llm(final_prompt)[0]["generated_text"]
        return answer, relevant_docs
