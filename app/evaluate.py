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
            You will be given a context and a question.
            Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
            Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

            Provide your answer as follows:

            Answer:::
            Evaluation: (your rationale for the rating, as a text)
            Total rating: (your rating, as a number between 1 and 5)

            You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

            Now here are the question and context.

            Question: {question}\n
            Context: {context}\n
            Answer::: """
    
        self.question_relevance_critique_prompt = """
            You will be given a question.
            Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
            Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

            Provide your answer as follows:

            Answer:::
            Evaluation: (your rationale for the rating, as a text)
            Total rating: (your rating, as a number between 1 and 5)

            You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

            Now here is the question.

            Question: {question}\n
            Answer::: """
        
        self.question_standalone_critique_prompt = """
            You will be given a question.
            Your task is to provide a 'total rating' representing how context-independent this question is.
            Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
            For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
            The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

            For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

            Provide your answer as follows:

            Answer:::
            Evaluation: (your rationale for the rating, as a text)
            Total rating: (your rating, as a number between 1 and 5)

            You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

            Now here is the question.

            Question: {question}\n
            Answer::: """

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
            # model="llama-3.3-70b-versatile",
            model="openai/gpt-oss-120b",
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
                assert len(answer) < 300, "Answer is too long"
                outputs.append(
                    {
                        "context": sampled_context.page_content,
                        "question": question,
                        "answer": answer,
                        "source_doc": sampled_context.metadata["source"],
                    }
                )
            except:
                continue
        return outputs

    def generate_evaluation_dataset(
        self,
    ):
        docs_processed = self.CHUNKS
        outputs = self.generate_qa(docs_processed)

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
            try:
                for criterion, evaluation in evaluations.items():
                    score, eval = (
                        int(evaluation.split("Total rating: ")[-1].strip()),
                        evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                    )
                    output.update(
                        {
                            f"{criterion}_score": score,
                            f"{criterion}_eval": eval,
                        }
                    )
            except Exception as e:
                continue
        
        # final step of producing the result
        generated_questions = pd.DataFrame.from_dict(outputs)
        generated_questions = generated_questions.loc[
            (generated_questions["groundedness_score"] >= 3)
            & (generated_questions["relevance_score"] >= 3)
            & (generated_questions["standalone_score"] >= 3)
        ]

        eval_dataset = datasets.Dataset.from_pandas(
            generated_questions, split="train", preserve_index=False
        )

        return eval_dataset
