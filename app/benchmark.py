from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from typing import Optional, List
from rerankers import Reranker
import json
import gc
import re
import glob
import pandas as pd
import torch
from evaluate import Evaluate
from reader import Reader
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
import os
from retriever import Retriever
from reader import Reader

class Benchmark:
    def __init__(
        self,
        chunkz,
        evaluator_name: str = "OLLAMA_llama3",
    ):
        self.EVALUATION_PROMPT = """
            ###Task Description:
            An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
            1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
            2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
            3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
            4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

            ###The instruction to evaluate:
            {instruction}

            ###Response to evaluate:
            {response}

            ###Reference Answer (Score 5):
            {reference_answer}

            ###Score Rubrics:
            [Is the response correct, accurate, and factual based on the reference answer?]
            Score 1: The response is completely incorrect, inaccurate, and/or not factual.
            Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
            Score 3: The response is somewhat correct, accurate, and/or factual.
            Score 4: The response is mostly correct, accurate, and factual.
            Score 5: The response is completely correct, accurate, and factual.

            ###Feedback:"""

        self.evaluation_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a fair evaluator language model."),
                HumanMessagePromptTemplate.from_template(self.EVALUATION_PROMPT),
            ]
        )
        
        self.evaluator_name = evaluator_name
        self._eval_chat_model = None
        self.chunk_size = 500
        self.embedder = "thenlper/gte-small"
        self.rerank = True
        self.chunks = chunkz
    
    @property
    def eval_chat_model(self): # replace with better evaluator chat model
        if self._eval_chat_model is None:
            self._eval_chat_model = ChatGroq(
            # model="llama-3.3-70b-versatile",
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        return self._eval_chat_model

    def run_rag_tests(
        self,
        eval_dataset,
        llm,
        knowledge_index: VectorStore,
        output_file: str = None,
        reranker: Optional[Reranker] = None,
        verbose: Optional[bool] = True,
        test_settings: Optional[str] = None,
    ):
        outputs = []
        ragas_outputs = []
        try:
            if output_file:
                with open(output_file, 'r') as f:
                    outputs = json.load(f)
        except:
            outputs = []
        
        for example in eval_dataset:
            question = example["question"]
            if question in [output["question"] for output in outputs]:
                continue

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            reader = Reader()
            answer, relevant_docs = reader.answer_with_rag(
                question, llm, knowledge_index,
                reranker=reranker,
                num_retrieved_docs=3,
                num_docs_final=2,
            )

            gc.collect()
            torch.cuda.empty_cache()

            if verbose:
                print("="*55)
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f'True answer: {example["answer"]}')
            result = {
                "question": question,
                "true_answer": example["answer"],
                "source_doc": example["source_doc"],
                "generated_answer": answer,
                "retrieved_docs": [
                    {
                        "page_content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                        "metadata": doc.metadata if hasattr(doc, "metadata") else {}
                    }
                    for doc in relevant_docs
                ],
            }
            if test_settings:
                result["test_settings"] = test_settings
            outputs.append(result)

            #UNCOMMENT FOR OTHER PURPOSE
            if output_file:
                with open(output_file, "w") as f:
                    json.dump(outputs, f)

            # modified for the ragas output purpose
            ragas_result = {
                "question": question,
                "true_answer": example["answer"],      # ground truth
                "generated_answer": answer,            # model answer
                "retrieved_docs": [
                    doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    for doc in relevant_docs
                ]
            }
            ragas_outputs.append(ragas_result)
        return ragas_outputs

    def evaluate_answers(
        self,
        answer_path: str,
        eval_chat_model,
        evaluator_name: str,
        evaluation_prompt_template: ChatPromptTemplate,
    ) -> None:
        answers = []
        if os.path.isfile(answer_path):
            try:
                with open(answer_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        answers = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse {answer_path}: {e}")
                return

        for experiment in answers:
            if f"eval_score_{evaluator_name}" in experiment:
                continue

            eval_prompt = evaluation_prompt_template.format_messages(
                instruction=experiment["question"],
                response=experiment["generated_answer"],
                reference_answer=experiment["true_answer"],
            )
            eval_result = eval_chat_model.invoke(eval_prompt)
            feedback, score = [
                item.strip() for item in eval_result.content.split("[RESULT]")
            ]
            experiment[f"eval_score_{evaluator_name}"] = score
            experiment[f"eval_feedback_{evaluator_name}"] = feedback

            with open(answer_path, "w") as f:
                json.dump(answers, f)

    def load_embeddings(
        self,
        langchain_docs: List[Document],
        chunk_size: int,
        embedding_model_name: Optional[str] = "thenlper/gte-small",
    ) -> FAISS:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            multi_process=False,
            # model_kwargs={"device": "cuda"},
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        index_name = (
            f"index_chunk:{chunk_size}_embeddings:{embedding_model_name.replace('/', '~')}"
        )
        index_folder_path = f"./data/indexes/{index_name}/"
        if os.path.isdir(index_folder_path):
            return FAISS.load_local(
                index_folder_path,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )

        else:
            print("Index not found, generating it...")
            knowledge_index = FAISS.from_documents(
                self.chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            knowledge_index.save_local(index_folder_path)
            return knowledge_index

    def safe_parse_score(
        self,
        x,
    ):
        try:
            match = re.search(r'\d+', str(x))
            return int(match.group()) if match else None
        except:
            return None

    def evaluate(
        self,
        eval_dataset,
        RAW_KNOWLEDGE_BASE,
        reader_llm=None,
        reranker: Optional[Reranker] = None,
        reader_model_name=None,
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if not os.path.exists("./output"):
            os.mkdir("./output")

        if reader_llm is None or reranker is None or reader_model_name is None:
            reader = Reader()
            READER_LLM = reader.reader_llm
            reranker = reader.reranker
            READER_MODEL_NAME = reader.READER_MODEL_NAME
        else:
            READER_LLM = reader_llm
            reranker = reranker
            READER_MODEL_NAME = reader_model_name

        settings_name = f"chunk:{self.chunk_size}_embeddings:{self.embedder.replace('/', '~')}_rerank:{self.rerank}_reader-model:{READER_MODEL_NAME.replace('/', '~')}"
        output_file_name = f"./output/rag_{settings_name}.json"
        print(f"RUNNING EVALUATION FOR {settings_name}:")
        print(f"RUNNING EVALUATION FOR THE FILENAME: {output_file_name}:")
        print("Loading knowledge base embeddings...")
        knowledge_index = self.load_embeddings(
            RAW_KNOWLEDGE_BASE,
            chunk_size=self.chunk_size,
            embedding_model_name=self.embedder,
        )
        print("Running RAG...")
        self.run_rag_tests(
            eval_dataset=eval_dataset,
            llm=READER_LLM,
            knowledge_index=knowledge_index,
            output_file=output_file_name,
            reranker=reranker,
            verbose=False,
            test_settings=settings_name,
        )
        print("RUNNING EVALUATION...")
        self.evaluate_answers(
            output_file_name,
            self.eval_chat_model,
            self.evaluator_name,
            self.evaluation_prompt_template,
        )
        gc.collect()
        torch.cuda.empty_cache()
        outputs = []
        for file in glob.glob("./output/*.json"):
            output = pd.DataFrame(json.load(open(file, "r")))
            output["settings"] = file
            outputs.append(output)
        try:
            result = pd.concat(outputs)
            result["eval_score_OLLAMA_llama3"] = result["eval_score_OLLAMA_llama3"].apply(self.safe_parse_score)
            average_scores = result.groupby("settings")["eval_score_OLLAMA_llama3"].mean()
            average_scores.sort_values()
            return average_scores
        except Exception as e:
            print(f"Error seen as {e}")
            return
