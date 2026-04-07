from retriever import Retriever
from reader import Reader
from benchmark import Benchmark

def main():
    retriever = Retriever()
    knowledge_index = retriever.build_vector_database()

    reader = Reader()
    answer, relevant_docs = reader.answer_with_rag(
        question = "What is the definition of basic remuneration?",
        llm=reader.reader_llm,
        knowledge_index=knowledge_index,
        reranker=reader.reranker,
    )

    print(answer)

    bench = Benchmark()
    average_scores = bench.evaluate()

    print(f"The average scores after the evaluation are: {average_scores}")

if __name__ == "__main__":
    main()
