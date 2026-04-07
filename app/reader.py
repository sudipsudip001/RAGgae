from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
import torch
from retriever import Retriever
from rerankers import Reranker
from transformers import Pipeline
from typing import Optional

class Reader:
    def __init__(self):
        self.READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
        self.prompt_in_chat_format = [
            {
                "role": "system",
                "content": """You are a precise information retrieval assistant. Your ONLY job is to extract exact information from the provided context.

        STRICT RULES — violations are not acceptable:
        - Answer EXCLUSIVELY from the provided context and only answer the given question. No external knowledge.
        - Do NOT paraphrase, infer, summarize, or elaborate.
        - Do NOT add any explanation, commentary, or filler phrases.
        - Do NOT combine or merge information from multiple sources unless the question explicitly asks for it.
        - If the answer is not explicitly stated in the context, respond ONLY with: "Not found in the provided context."
        - Copy the Source and Page number exactly as they appear in the context — no modifications.
        - Every claim in your answer must be backed by a direct quote.

        """
            },
            {
                "role": "user",
                "content": """Context:
        {context}

        Question: {question}

        Respond using this exact format and nothing else:

        If answerable:
            Answer: <one or two sentences maximum, strictly from the context>

            Evidence:
            - Source | Page
            exact context

        If unanswerable: write only "Not found in the provided context." with no Evidence section."""
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
            self._reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
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


    def ask_the_reader(self):
        retriever = Retriever()
        KNOWLEDGE_VECTOR = retriever.build_vector_database()
        retrieved_docs = retriever.retrieve_docs(KNOWLEDGE_VECTOR, "What is the meaning of 'Basic remuneration'?") #-----------> missing something here?

        retrieved_docs_text = [
            doc.page_content for doc in retrieved_docs
        ]
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
        )
        final_prompt = self.RAG_PROMPT_TEMPLATE.format(
            question="What is the meaning of 'Basic remuneration'?", context=context
        )

        answer = self.READER_LLM(final_prompt)[0]["generated_text"]
        return answer

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

        context = "\nExtracted documents:\n"
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")

            context += f"""
                --- Document {i} ---
                Source: {source}
                Page: {page}

                {doc.page_content}
            """
        final_prompt = self.RAG_PROMPT_TEMPLATE.format(question=question, context=context)
        print("=> Generating answer...")
        answer = llm(final_prompt)[0]["generated_text"]

        return answer, relevant_docs
