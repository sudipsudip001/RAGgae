from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from retriever import Retriever
from benchmark import Benchmark
from dotenv import load_dotenv
from pydantic import BaseModel
from evaluate import Evaluate
from ingest import Ingestor
from ragged import Ragged
from reader import Reader
import torch
import gc
import os

load_dotenv()

ingestor = None
retriever = None
reader = None
VECTOR_DB = None
RAW_KNOWLEDGE_BASE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestor, retriever, reader, VECTOR_DB, RAW_KNOWLEDGE_BASE

    print("Loading models...")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"   # skip network checks after first download

    ingestor = Ingestor(
        ["../docs/crime_act.pdf", "../docs/interim_government_act.pdf", "../docs/labor_act.pdf"],
        200, 50, "thenlper/gte-small",
    )
    VECTOR_DB = ingestor.vector_database
    RAW_KNOWLEDGE_BASE = ingestor.raw_knowledge_base

    retriever = Retriever(VECTOR_DB)

    reader = Reader(
        "HuggingFaceH4/zephyr-7b-beta",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    print("All models loaded — server ready!")

    yield

    print("Shutting down...")
    gc.collect()
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

class AnswerResponse(BaseModel):
    answer: str
    found_in_context: bool

class EvalResponse(BaseModel):
    llm_scores: list
    ragas_scores: list

#------------------------------------ENDPOINTS------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": reader is not None}

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    result, relevant_docs = reader.answer_with_rag(
        question=req.question,
        llm=reader.reader_llm,
        knowledge_index=VECTOR_DB,
        reranker=reader.reranker,
    )

    if torch.cuda.is_available():
        used = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        if used / total > 0.9:
            gc.collect()
            torch.cuda.empty_cache()

    return AnswerResponse(
        answer=result["answer"],
        found_in_context=result["found_in_context"]
    )

@app.post("/evaluate", response_model=EvalResponse)
def evaluate():
    gc.collect()
    torch.cuda.empty_cache()

    chunks_data = ingestor.chunker()
    evaluator = Evaluate(chunks_data)
    eval_dataset = evaluator.generate_evaluation_dataset()

    benchmarker = Benchmark(evaluator_name="OLLAMA_llama3", chunkz=chunks_data)
    scores = benchmarker.evaluate(
        eval_dataset=eval_dataset,
        RAW_KNOWLEDGE_BASE=RAW_KNOWLEDGE_BASE,
        reader_llm=reader.reader_llm,
        reranker=reader.reranker,
        reader_model_name=reader.READER_MODEL_NAME,
    )

    outputs = benchmarker.run_rag_tests(
        eval_dataset=eval_dataset,
        knowledge_index=VECTOR_DB,
        llm=reader.reader_llm,
    )
    ragas_dataset = [
        {
            "question": item["question"],
            "ground_truth": item["true_answer"],
            "answer": item["generated_answer"]["answer"] if isinstance(item["generated_answer"], dict) else item["generated_answer"],
            "contexts": item["retrieved_docs"]
        }
        for item in outputs
    ]
    if len(ragas_dataset) == 0:
        print(f"Skipping RAGAS, dataset is empty. Fix QA generation first.")
    else:
        raga = Ragged(ragas_dataset, "thenlper/gte-small", "mistralai/Mistral-7B-Instruct-v0.3")
        final_score = raga.score()
        return EvalResponse(
            llm_scores=scores,
            ragas_scores=final_score.to_dict(orient="records")
        )
