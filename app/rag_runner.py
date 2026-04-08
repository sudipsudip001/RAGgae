from retriever import Retriever
from benchmark import Benchmark
from evaluate import Evaluate
from ingest import Ingestor
from reader import Reader
from ragged import Ragged
from dotenv import load_dotenv
import torch
import gc

def main():
    ingestor = Ingestor(
        ["../docs/crime_act.pdf", "../docs/interim_government_act.pdf", "../docs/labor_act.pdf"],
        200,
        50,
        "thenlper/gte-small"
    )
    VECTOR_DB = ingestor.vector_database
    RAW_KNOWLEDGE_BASE = ingestor.raw_knowledge_base

    retriever = Retriever(
        VECTOR_DB
    )
    reader = Reader(
        "HuggingFaceH4/zephyr-7b-beta",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    print("RAG system loaded! Type 'exit/quit/q' to quit and 'evaluate' to evaluate the model.\n")
    while True:
        question = input("Ask a question: ").strip()

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
        print(relevant_docs[:500])
        print("----------------------------------")

        gc.collect()
        torch.cuda.empty_cache()

        if question.lower() in ("evaluate", "eval"):
            chunks_data = ingestor.chunker()
            evaluator = Evaluate(chunks_data)

            eval_dataset = evaluator.generate_evaluation_dataset()

            benchmarker = Benchmark(
                evaluator_name="OLLAMA_llama3",
                chunkz=chunks_data
            )

            print("--------------LLM as a judge-------------------")
            scores = benchmarker.evaluate(
                eval_dataset=eval_dataset,
                RAW_KNOWLEDGE_BASE=RAW_KNOWLEDGE_BASE,
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
                    "answer": item["generated_answer"],
                    "contexts": item["retrieved_docs"]
                }
                for item in outputs
            ]

            # RAGAS evaluation
            raga = Ragged(
                ragas_dataset,
                "thenlper/gte-small",
                "mistralai/Mistral-7B-Instruct-v0.3",
            )
            final_score = raga.score()
            print(final_score)

if __name__ == "__main__":
    load_dotenv()
    main()
