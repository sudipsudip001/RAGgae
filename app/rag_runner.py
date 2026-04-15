from retriever import Retriever
from benchmark import Benchmark
from evaluate import Evaluate
from ingest import Ingestor
from reader import Reader
from ragged import Ragged
from dotenv import load_dotenv
import torch
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv()

ingestor = Ingestor(
    ["../docs/crime_act.pdf", "../docs/interim_government_act.pdf", "../docs/labor_act.pdf"],
    200,
    50,
    "thenlper/gte-small"
)

VECTOR_DB = ingestor.vector_database
RAW_KNOWLEDGE_BASE = ingestor.raw_knowledge_base

print("Loading retriever...")
retriever = Retriever(VECTOR_DB)
print("Loading reader (LLM + reranker)...")
reader = Reader(
    "HuggingFaceH4/zephyr-7b-beta",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
print("All models loaded")

def run_qa_loop():
    while True:
        question = input("Ask a question, type 'exit/quit/q' to quit.\n").strip()

        if not question:
            print("Enter a valid question.")
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # docs_read = retriever.retrieve_docs(question)

        answer, relevant_docs = reader.answer_with_rag(
            question = question,
            llm=reader.reader_llm,
            knowledge_index=VECTOR_DB,
            reranker=reader.reranker,
        )

        print("RAG ANSWER:")
        print("----------------------------------")
        print(answer)
        print("----------------------------------")
        print("DOCS REFERENCED:")
        print("----------------------------------")
        for i, doc in enumerate(relevant_docs):
            print("----------------------------------")
            print(f"Document {i}")
            print(doc.page_content[:100])
            print(doc.metadata)
        print("----------------------------------")

        gc.collect()
        torch.cuda.empty_cache()

def run_evaluation():
    print("-------------Evaluating the model----------------")
    chunks_data = ingestor.chunker()
    evaluator = Evaluate(chunks_data)

    eval_dataset = evaluator.generate_evaluation_dataset()
    print("Completed generating evaluation dataset")

    print(f"THE LENGTH OF EVAL_DATASET IS: {len(eval_dataset)}")


    benchmarker = Benchmark(
        evaluator_name="OLLAMA_llama3",
        chunkz=chunks_data,
    )


    
    print("--------------LLM as a judge-------------------")
    scores = benchmarker.evaluate(
        eval_dataset=eval_dataset,
        RAW_KNOWLEDGE_BASE=RAW_KNOWLEDGE_BASE,
        reader=reader,
    )
    print("-------------LLM scores-------------")
    print("Scores from LLM are: ", scores)







    print("--------------RAGAS work-------------------")

    outputs = benchmarker.run_rag_tests(
        eval_dataset=eval_dataset,
        knowledge_index=VECTOR_DB,
        llm=reader.reader_llm,
    ) #will work if run_rag_tests() works properly
    ragas_dataset = [
        {
            "question": item["question"],
            "ground_truth": item["true_answer"],
            "answer": item["generated_answer"]["answer"] if isinstance(item["generated_answer"], dict) else item["generated_answer"],
            "contexts": item["retrieved_docs"]
        }
        for item in outputs
    ]

    # RAGAS evaluation
    print(f"Dataset length: {len(ragas_dataset)}")
    print(f"First row sample: {ragas_dataset[0] if ragas_dataset else 'EMPTY!'}")


    raga = Ragged(
        ragas_dataset,
        "thenlper/gte-small",
        "mistralai/Mistral-7B-Instruct-v0.3",
    )
    print("THE REAL WORK STARTS FROM HERE")

    if len(ragas_dataset) == 0:
        print(f"Skipping RAGAS, dataset is empty. Fix QA generation first.")
    else:
        final_score = raga.score()
        print(final_score)
        final_score.to_csv("ragas_evaluation_metrics.csv", encoding='utf-8', index=False)
        exit()

def main():
    while True:
        print("\nWhat do you want to do?")
        print("   1. Q&A")
        print("   2. Run evaluation")
        print("   3. Exit")
        choice = input("Choice: ").strip()

        if choice == "1":
            run_qa_loop()
        elif choice == "2":
            run_evaluation()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    load_dotenv()
    main()
