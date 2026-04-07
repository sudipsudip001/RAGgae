from retriever import Retriever
from reader import Reader
from benchmark import Benchmark
import torch
from dotenv import load_dotenv
import gc
from ingest import Ingestor

def main():
    ingestor = Ingestor(
        ["../docs/crime_act.pdf", "../docs/interim_government_act.pdf", "../docs/labor_act.pdf"],
        200,
        50,
        "thenlper/gte-small"
    )
    VECTOR_DB = ingestor.vector_database

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
        print(relevant_docs)
        print("----------------------------------")

        gc.collect()
        torch.cuda.empty_cache()

    # bench = Benchmark()
    # average_scores = bench.evaluate()
    # print(f"The average scores after the evaluation are: {average_scores}")

if __name__ == "__main__":
    load_dotenv()
    main()
