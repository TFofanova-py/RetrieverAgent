import asyncio

from httpcore import stream
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
import json
from langchain.chains import create_history_aware_retriever
from typing import List
from langchain_core.documents import Document
import logging
from datetime import datetime
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, HumanMessage
import argparse
from make_index.setup_opensearch_db import process_chunk


logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/agent.log")]
    )
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, model: str, verbose: bool = False):
        self.docs: List[Document] = []
        self.model = model
        self.verbose = verbose
        self.answer: str = ""
        self.messages: List[BaseMessage] = []
        self.embedder = OllamaEmbeddings(model="llama3.1")
        self.llm = ChatOllama(model=model, temperature=0.)
        self.db_kwargs = json.load(open("creds.json", "rb"))["OPEN_SEARCH_KWARGS"]
        self.db_kwargs["http_auth"] = (
            self.db_kwargs.get("http_auth", {}).get("login"),
            self.db_kwargs.get("http_auth", {}).get("password")
        )
        self.db = OpenSearchVectorSearch(
            embedding_function=self.embedder,
            **self.db_kwargs
        )
        self.index_name = self.db_kwargs.get("index_name")
        self.retriever = self.db.as_retriever()
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        rag_prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = rag_prompt | self.add_lang_prompt | self.llm | StrOutputParser()
        logger.info("Init agent")

    @staticmethod
    def add_lang_prompt(inputs):
        inputs.messages = [SystemMessage("Always return output in German.")] + inputs.messages
        return inputs

    async def has_save_tag(self) -> bool:
        return self.messages[-1].content.strip().lower().startswith("#save")

    async def create_answer(self):
        # is_query_task = asyncio.create_task(self.llm.ainvoke(f"I give you a text. Your task is to decide if the text is a query for information. Output must be just yes or no.\nText: {self.messages[-1].content}"))
        # ret_task = asyncio.create_task(self.retriever.ainvoke(self.messages[-1].content))
        # ret_task = asyncio.create_task(self.history_aware_retriever.ainvoke({"input": state["messages"][-1].content, "chat_history": state["messages"][-6:]}))
        # llm_task = asyncio.create_task(self.llm.astream([SystemMessage(content="Always return output in German."), self.messages[-1]]))

        # is_query = (await is_query_task).content.lower().strip()

        # if any([is_query.startswith(x) for x in ["no", "nein"]]):
        #     logger.info("Classificator: it's not a query. LLM answers")
        #     # ret_task.cancel()
        #     async for chunk in self.llm.astream([SystemMessage(content="Always return output in German."), self.messages[-1]]):
        #         yield chunk
        # else:
        #     logger.info("Classificator: it's a query. Retriever answers")
        self.docs = await self.retriever.ainvoke(self.messages[-1].content)  # ret_task
        logger.info(f"Retriever documents: {self.docs}")

        ret_answer = await asyncio.create_task(
            self.rag_chain.ainvoke({"context": self.docs, "question": self.messages[-1]}))
        logger.info(f"Retrieved: {ret_answer}")

        grade = await self.llm.ainvoke(
            f"I give you a text of answer. Your task is to decide if the text means \"I don't know\". Output must be just yes or no.\nText: {ret_answer}")
        grade_content = grade.content.lower().strip()
        logger.info(f"Grading answer, {grade_content}")

        if any([grade_content.startswith(x) for x in ["no", "nein"]]):
            # llm_task.cancel()
            yield ret_answer
        else:
            async for chunk in self.llm.astream([SystemMessage(content="Always return output in German."), self.messages[-1]]):
                yield chunk

    async def save_answer(self):
        logger.info("Save answer")
        self.messages.append(AIMessage(self.answer))


    async def save_information(self):
        logger.info("Save information")
        text, embs, metadata = await process_chunk(self.messages[-1].content.replace("#save", ""),
                                                   "chat_message",
                                                   datetime.now())
        text_emb_data = [(text, emb) for emb in embs if emb]
        logger.info("Adding embeddings")
        self.db.add_embeddings(text_emb_data,
                          metadatas=[metadata] * len(embs),
                          index_name=self.index_name
                          )
        self.messages.append(AIMessage(f"Informationen werden im Index {self.index_name} gespeichert"))



async def main_gr():
    import gradio as gr


    async def chat_response(input_message, chat_history):
        start = datetime.now()
        logger.info(f"Input message: {input_message}")
        agent.messages.append(HumanMessage(input_message))
        msg = ""
        async for chunk in agent.create_answer():
            msg += chunk if isinstance(chunk, str) else chunk.content
            yield msg

        agent.messages.append(AIMessage(msg))
        logger.info(f"Response time: {datetime.now() - start}")


    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", default="llama3", help="Model to use")
    #
    # args = parser.parse_args()

    agent =  Agent(model="llama3.1", verbose=False)
    logger.info("Agent started")

    with gr.Blocks() as demo:
        chatbot = gr.ChatInterface(fn=chat_response,
                                   type="messages",
                                   chatbot=gr.Chatbot(type="messages", value=[{"role": "assistant", "content": "Hallo! Ich bin Ihr ISMS-Assistent. Stellen Sie Fragen"}]),
                                   textbox=gr.Textbox(placeholder="Type a message...", label="Your Message")
                                   )
    demo.launch(share=True)


if __name__ == "__main__":
    asyncio.run(main_gr())
