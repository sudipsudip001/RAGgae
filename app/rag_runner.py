from retriever import Retriever
from reader import Reader
from benchmark import Benchmark
import torch
import gc

def main():
    retriever = Retriever()
    knowledge_index = retriever.build_vector_database()
    reader = Reader()

    print("RAG system loaded! Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if not question:
            print("Enter a valid question.")
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        answer, relevant_docs = reader.answer_with_rag(
            question = question,
            llm=reader.reader_llm,
            knowledge_index=knowledge_index,
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
    main()
