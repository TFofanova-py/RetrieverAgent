import pandas as pd
from sqlalchemy.dialects.mssql.information_schema import columns

from agent import Agent
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, HumanMessage
import asyncio
import logging

logger = logging.getLogger(__name__)

def read_questions(file_path):
    with open(file_path) as f:
        questions = f.readlines()
    return [q.strip().split(",") for q in questions]

async def main():
    df = pd.DataFrame(columns=["topic", "question", "answer"])
    df_questions = pd.read_csv("audit_questions.txt")
    df_questions.columns = ["topic", "question"]
    agent = Agent(model="gemma2", verbose=False)

    for i, row in df_questions.iterrows():
        print(f"Question ({i}) - Topic ({row['topic']}):", row["question"])
        agent.messages = [SystemMessage("Answer as fully and concrete as you can. You don't ask any questions! Output MUST BE in German."),
                          HumanMessage(row["question"]),]
        print("Answer: ", end="")
        answer = ""
        async for chunk in agent.create_answer(topic=row["topic"]):
            answer += chunk if isinstance(chunk, str) else chunk.content
        print(answer.strip())
        print("\n____________________________________________________________________________________")
        df = pd.concat([df, pd.DataFrame.from_dict({"question": [row["question"]], "answer": [answer]})], )
        if i % 5 == 0:
            df.to_csv("../qa_dataset.csv", index=False)

    df.to_csv("../qa_dataset.csv", index=False)

if __name__ == '__main__':
    asyncio.run(main())