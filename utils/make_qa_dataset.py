import pandas as pd
from pydantic import BaseModel
from agent import Agent
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, HumanMessage
import asyncio
import ollama
import re
from make_index.setup_opensearch_db import get_chunks
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def read_questions(file_path):
    with open(file_path) as f:
        questions = f.readlines()
    return [q.strip() for q in questions]

async def main():
    df = pd.DataFrame(columns=["question", "answer"])
    questions = read_questions("../questions.txt")
    agent = Agent(model="gemma2", verbose=False)

    for i, q in enumerate(questions):
        print(f"Question ({i}):", q)
        agent.messages = [SystemMessage("Answer as fully and concrete as you can. You don't ask any questions! Output MUST BE in German."),
                          HumanMessage(q)]
        print("Answer: ", end="")
        answer = ""
        async for chunk in agent.create_answer():
            answer += chunk if isinstance(chunk, str) else chunk.content
        print(answer.strip())
        print("\n____________________________________________________________________________________")
        df = pd.concat([df, pd.DataFrame.from_dict({"question": [q], "answer": [answer]})], )
        if i % 5 == 0:
            df.to_csv("../qa_dataset.csv", index=False)

    df.to_csv("../qa_dataset.csv", index=False)
# ---------------------------------------------

class Pair(BaseModel):
    question: str
    answer: str

chunk_prompt = """
Your task is to write 7 question-answer pairs to the text below, which you can ask the ISMS (information security management system) assistant. Ask questions from the perspective of a company that needs to apply this law in its daily operations. Answer should be as full as possible and use information from the text, long at least 10 sentences.
Language of output is German.
Output must be a list of pairs without any additional thoughts and explanations.
Examples:
- {"question": "Was bedeutet NIS2 für mein Unternehmen?", "answer": "...(some answer)"}
- {"question": "Welche Produkte und Services muss ich einsetzen, um NIS2 gerecht zu werden?", "answer": "..."}
- {"question": "Welche Prozesse müssen gemäß NIS2 in meinem Unternehmen vorhanden sein?", "answer": "..."}
- {"question": "Welche Auswirkungen hat NIS2 auf mich, als einen Mitarbeiter der IT Abteilung?", "answer": "..."}
- {"question": "Welche Auswirkungen hat NIS2 auf das Unternehmen, in dem ich beschäftigt bin?", "answer": "..."}
"""

async def make_qa_examples(chunk, model: str = "llama3"):

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": chunk_prompt},
            {"role": "user", "content": chunk}]
    )
    response_content = response["message"]["content"]
    pairs = [Pair.model_validate_json(x.strip())
                          for x in re.findall(r"\{.+\}", response_content)]
    return pairs

async def main_pure_llama():
    df = pd.DataFrame(columns=["question", "answer"])
    paths = [
        ("../Docs/240722_nis2-regierungsentwurf.pdf", datetime(year=2024, month=7, day=22)),
        # ("../Docs/CELEX_02022L2555-20221227_DE_TXT.pdf", datetime(year=2022, month=12, day=27)),
        # ("../Docs/Leitfaden_zur_Basis-Absicherung.pdf", datetime(year=2020, month=11, day=1)),
        # ("../Docs/standard_200_1.pdf", datetime(year=2020, month=11, day=1)),
        # ("../Docs/standard_200_2.pdf", datetime(year=2020, month=11, day=2)),
        # ("../Docs/it-sicherheitsgesetz.pdf", datetime(year=2020, month=11, day=1)),
    ]

    for source, dt, chunks in get_chunks(paths=paths, chunk_size=2000):
        logger.info(f"Splitting {source} for chunks, OK")

        for i, chunk in enumerate(chunks[:15]):
            pairs = await make_qa_examples(chunk)
            logger.info(f"Processing chunk {i}, OK")
            df = pd.concat([df, pd.DataFrame.from_dict({"question": [x.question for x in pairs], "answer": [x.answer for x in pairs]})], ignore_index=True )
            if i % 5 == 0:
                df.to_csv("../qa_llama_dataset.csv", index=False)

        df.to_csv("../qa_llama_dataset.csv", index=False)


if __name__ == '__main__':
    asyncio.run(main())