import asyncio
from asyncio import sleep
import pandas as pd
from gradio.components.chatbot import ChatMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
import json
from langchain.chains import create_history_aware_retriever
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging
from datetime import datetime
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, HumanMessage
import argparse

from agents.retriever import retriever
from make_index.setup_opensearch_db import process_chunk


logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/agent.log")]
    )
logger = logging.getLogger(__name__)

TOPICS = json.load(open("creds.json", "rb"))["TOPICS"]


class Agent:
    def __init__(self, model: str, verbose: bool = False):
        self.docs: List[Document] = []
        self.model = model
        self.verbose = verbose
        self.answer: str = ""
        self.messages: List[BaseMessage] = []
        self.embedder = OllamaEmbeddings(model="llama3.1")
        self.llm = ChatOllama(model=model, temperature=.0)  # OllamaLLM(model=model, temperature=0.)
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
        self.retriever = self.db.as_retriever()  # search_kwargs={"k": 6}
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
        inputs.messages = [SystemMessage("ALWAYS return output in German. Search for relevant information from the documents but avoid describing these documents. Answer as fully as possible.")] + inputs.messages
        return inputs

    async def has_save_tag(self) -> bool:
        return self.messages[-1].content.strip().lower().startswith("#save")

    async def create_answer(self, topic: str = "", question: str = ""):
        logger.info(f"User input: {question}")
        if topic:
            self.docs = self.db.similarity_search_with_score(
                query=question,
                space_type='cosineSimilarity',
                efficient_filter={'bool': {'must': [{'term': {'metadata.type': 'answer'}}, {'regexp': {'metadata.topic': f'.*{topic[-5:]}'}}]}},  #  {'term': {'metadata.topic': topic}} - don't work
                k=3,
            )
        else:
            self.docs = await self.retriever.ainvoke(question)
        logger.info(f"Retriever documents: {self.docs}")

        ret_answer = await asyncio.create_task(
            self.rag_chain.ainvoke({"context": self.docs, "question": question}))
        logger.info(f"Retrieved: {ret_answer}")

        grade = await self.llm.ainvoke(
            f"I give you a text of answer. Your task is to decide if the text means \"I don't know\". Output must be just yes or no.\nText: {ret_answer}")
        grade_content = grade.content.lower().strip()  # grade.content.lower().strip()
        logger.info(f"Grading answer, {grade_content}")

        if any([grade_content.startswith(x) for x in ["no", "nein"]]):
            for chunk in ret_answer.split(" "):
                await sleep(0.2)
                yield chunk + " "
        else:
            async for chunk in self.llm.astream([SystemMessage(content="Always return output in German."), self.messages[-1]]):
                yield chunk

    async def create_answer_llm_first(self, topic: str = "", question: str = ""):
        logger.info(f"User input: {question}")
        llm_answer = await self.llm.ainvoke([SystemMessage(content="Always return output in German."), HumanMessage(content=question)])
        if topic:
            self.docs = self.db.similarity_search_with_score(
                query=question,
                space_type='cosineSimilarity',
                efficient_filter={'bool': {'must': [{'term': {'metadata.type': 'answer'}},
                                                    {'regexp': {'metadata.topic': f'.*{topic[-5:]}'}}]}},
                k=3,
            )
        else:
            self.docs = await self.retriever.ainvoke(question)
        prompt = f"""
        You'll provide a llm answer for a user query and list of documents.
        Your task is to improve the llm answer using texts of documents if they are relevant.
        Answer format and language must leave the same.
        Query: {question}
        Llm answer: {llm_answer.content}
        Documents: {'\n'.join([x[0].page_content for x in self.docs])}
        """
        async for chunk in self.llm.astream(prompt):
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

    async def grade_answer(self, topic: str = ""):
        question = self.messages[-2].content
        answer = self.messages[-1].content
        self.docs = self.db.similarity_search_with_score(
            query=question,
            space_type='cosineSimilarity',
            efficient_filter={'bool': {
                'must': [{'term': {'metadata.type': 'answer'}}, {'regexp': {'metadata.topic': f'.*{topic[-5:]}'}}]}},
            k=2,
        )
        logger.info(f"Grade answer, question: {question}, answer: {answer}")
        async for chunk in self.llm.astream([SystemMessage(content="You are a helpful assistant that always responds in German."),
                                             HumanMessage(
                                                 content=f"I give you the texts of question and answer. Your task is to grade the answer from 0 to 10 (be strict but polite) and write a report in a sheet-sandwich format with all details of your thoughts during the grading."
                                                         f"When grading, rely on source documents: {'\n'.join([x[0].page_content for x in self.docs])}"
                                                         f"Question:\n{question}"
                                                         f"Answer:\n{answer}"
                                                         f"Don't provide an example of the best answer. The report entirely MUST be in German")]):
            yield chunk


async def main_gr():
    import gradio as gr

    df_questions = pd.read_csv("audit_questions.txt")
    TOPICS = df_questions.value_counts("topic")[df_questions.value_counts("topic")>5].index.tolist()

    async def chat_response(input_message, chat_history, state, topic):
        start = datetime.now()

        logger.info(f"Input message: {input_message}, step: {state.get("step")}")
        agent.messages.append(HumanMessage(input_message))
        msg = "**Musterantwort**: "

        async for chunk in agent.create_answer_llm_first(question=agent.messages[-2].content, topic=topic):
            msg += chunk + " " if isinstance(chunk, str) else chunk.content
            yield msg
        agent.messages.append(AIMessage(msg))

        msg += f"\n\n**{state.get('curr_question') + 1} Frage**: "
        yield msg
        new_question = await ask_question(state, topic)
        for chunk in new_question.split(" "):
            msg += chunk + " "
            yield msg
            await sleep(0.2)
        agent.messages.append(AIMessage(new_question))

        logger.info(f"Response time: {datetime.now() - start}")


    async def ask_question(state, topic):
        idx = state["curr_question"]
        state["curr_question"] += 1  # TODO: handle end of list
        msg = df_questions[df_questions["topic"] == topic]["question"].to_list()[idx]
        return msg


    async def change_topic(new_value: str):
        new_state = {"step": 0, "curr_question": 0}
        question = await ask_question(new_state, new_value)
        return ([{"role": "assistant", "content": f"Hallo! Ich bin ein KI-Assistent für Informationssicherheitsaudits. Sie haben Abschnitt {new_value} ausgewählt"},
                 {"role": "assistant", "content": f"**1 Frage**: {question}"}],
                new_state)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", default="llama3", help="Model to use")
    #
    # args = parser.parse_args()

    agent =  Agent(model="gemma2", verbose=False)
    logger.info("Agent started")

    with gr.Blocks() as demo:
        topic_list = gr.Dropdown(label="Select a topics", choices=TOPICS, value=TOPICS[0], interactive=True)
        user_state = gr.State(value={"curr_question": 0})
        question = await ask_question(user_state.value, topic_list.value)
        messages = [AIMessage(f"Hallo! Ich bin ein KI-Assistent für Informationssicherheitsaudits. Sie haben Abschnitt {topic_list.value} ausgewählt"),
                    AIMessage(f"**1 Frage**: {question}")
                    ]
        agent.messages = messages
        chatbot = gr.ChatInterface(fn=chat_response,
                                   type="messages",
                                   chatbot=gr.Chatbot(type="messages", value=[ChatMessage(role="assistant", content=m.content) for m in messages]),
                                   textbox=gr.Textbox(placeholder="Type a message...", label="Your Message"),
                                   additional_inputs=[user_state, topic_list]
                                   )


        topic_list.change(fn=change_topic, inputs=topic_list, outputs=[chatbot.chatbot, user_state])
    demo.launch(share=True)


if __name__ == "__main__":
    asyncio.run(main_gr())
