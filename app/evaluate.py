from langchain_core.documents import Document
from typing import List, Dict
from openai import OpenAI
import pandas as pd
import datasets
import random
import os

class Evaluate:
    def __init__(
        self,
        chunkz,
    ) -> None:
        self._client = None

        self.N_GENERATIONS = 10

        self.QA_generation_prompt = """
        Your task is to write a factoid question and an answer given a context.
        Your factoid question should be answerable with a specific, concise piece of factual information from the context.
        Your factoid question should be formulated in the same style as questions users could ask in a search engine.
        This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

        Provide your answer as follows:

        Output:::
        Factoid question: (your factoid question)
        Answer: (your answer to the factoid question)

        Now here is the context.

        Context: {context}\n
        Output:::"""

        self.question_groundedness_critique_prompt = """
        You are evaluating a question-answer dataset.

        Given the context and question, rate how well the question is grounded in the context.

        Evaluation criteria:
        5 = answer clearly supported by context
        4 = mostly supported
        3 = partially supported
        2 = weak support
        1 = not supported

        Respond ONLY in this format:

        Evaluation: <short reasoning>
        Total rating: <1-5>

        Context:
        {context}

        Question:
        {question}
        """
    
        self.question_relevance_critique_prompt = """
            Evaluate whether the question is relevant to the given document collection.

            A relevant question:
            - relates to the information in the document
            - could be answered using the document

            5 = highly relevant
            1 = completely irrelevant

            Respond ONLY in this format:

            Evaluation: <short reasoning>
            Total rating: <1-5>

            Question:
            {question}
            """
        
        self.question_standalone_critique_prompt = """
        Evaluate whether the question is understandable without additional context.

        5 = fully self-contained
        1 = unclear without context

        Respond ONLY in this format:

        Evaluation: <short reasoning>
        Total rating: <1-5>

        Question:
        {question}
        """

        self.CHUNKS = chunkz

    @property
    def client(self): # replace with better client
        if self._client is None:
            self._client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        return self._client

    def call_llm(self, client_instance: OpenAI, prompt: str, system_prompt: str = None):
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = client_instance.chat.completions.create(
            messages=messages,
            # model="openai/gpt-oss-120b",
            model="moonshotai/kimi-k2-instruct-0905",
            # model="llama-3.3-70b-versatile",
            max_tokens=200,
            temperature=0,
            top_p=0.9,
        )
        return response.choices[0].message.content

    def generate_qa(
        self,
        docs_processed: List[Document]
    ) -> List[Dict]:
        outputs = []
        for sampled_context in random.sample(docs_processed, self.N_GENERATIONS):
            output_QA_couple = self.call_llm(
                self.client, self.QA_generation_prompt.format(context=sampled_context.page_content)
            )
            try:
                question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
                answer = output_QA_couple.split("Answer: ")[-1]
                assert len(answer) < 500, "Answer is too long"
                outputs.append(
                    {
                        "context": sampled_context.page_content,
                        "question": question,
                        "answer": answer,
                        "source_doc": sampled_context.metadata["source"],
                    }
                )
            except Exception as e:
                print(f"Skipping due to error: {e}")
                continue
        return outputs

    def generate_evaluation_dataset(
        self,
    ):
        docs_processed = self.CHUNKS
        outputs = self.generate_qa(docs_processed)
        # print(f"The outputs from the QnA looks something like this: {outputs}")
        print("-"*7, "Generating evaluation dataset", "-"*7)

        for output in outputs:
            evaluations = {
                "groundedness": self.call_llm(
                    self.client,
                    self.question_groundedness_critique_prompt.format(
                        context=output["context"], question=output["question"]
                    ),
                ),
                "relevance": self.call_llm(
                    self.client,
                    self.question_relevance_critique_prompt.format(question=output["question"]),
                ),
                "standalone": self.call_llm(
                    self.client,
                    self.question_standalone_critique_prompt.format(question=output["question"]),
                ),
            }

            for criterion, evaluation in evaluations.items():
                try:
                    score_raw = evaluation.split("Total rating:")[-1].strip()
                    score = int(score_raw.split()[0].rstrip(".,\n"))

                    eval_text = evaluation.split("Evaluation:")[-1].split("Total rating:")[0].strip()

                    output.update({
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval_text,
                    })
                except Exception as e:
                    print(f"Parse error for [{criterion}]: {e}")
                    print(f"  Raw response: {evaluation[:150]}")
                    # Set a default failing score so the row isn't silently dropped
                    output.update({
                        f"{criterion}_score": 0,
                        f"{criterion}_eval": "parse_failed",
                    })
        
        # final step of producing the result
        generated_questions = pd.DataFrame.from_dict(outputs)
        generated_questions = generated_questions.loc[
            (generated_questions["groundedness_score"] >= 3)
            & (generated_questions["relevance_score"] >= 3)
            & (generated_questions["standalone_score"] >= 3)
        ]

        # Add this before the .loc[] filter
        print("Score summary:")
        for _, row in generated_questions.iterrows():
            print(f"  Q: {row['question'][:60]}")
            print(f"     groundedness={row.get('groundedness_score')} | relevance={row.get('relevance_score')} | standalone={row.get('standalone_score')}")


        eval_dataset = datasets.Dataset.from_pandas(
            generated_questions, split="train", preserve_index=False
        )

        return eval_dataset
