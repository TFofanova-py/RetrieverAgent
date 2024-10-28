import asyncio
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.tools.tavily_search import TavilySearchResults
import json
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langchain.chains import create_history_aware_retriever
from typing import List, Literal, Tuple
from langchain_core.documents import Document
import logging
from datetime import datetime
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
import argparse
from make_index.setup_opensearch_db import process_chunk


# _set_env("TAVILY_API_KEY")
# web_search_tool = TavilySearchResults(k=3)

logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/agent.log")]
    )
logger = logging.getLogger(__name__)


class State(MessagesState):
    docs: List[Document]
    answer: str
    verbose: bool


def add_lang_prompt(inputs):
    inputs.messages = [SystemMessage("Always return output in German.")] + inputs.messages
    return inputs


async def setup_agent(model: str, verbose: bool = False) -> Tuple[CompiledStateGraph, dict]:
    async def has_save_tag(state: State) -> Literal["save_information", "set_initial_state"]:
        if state["messages"][-1].content.strip().lower().startswith("#save"):
            return "save_information"
        return "set_initial_state"

    def set_initial_state(state: State):
        logger.info("Set initial state")
        state["verbose"] = verbose
        state["docs"] = []
        return state

    async def cls_query(state: State) -> Literal["call_model", "retrieve"]:
        is_query_response = await llm.ainvoke(
            f"I give you a text. Your task is to decide if the text is a query for information. Output must be just yes or no.\nText: {state["messages"][-1].content}")
        is_query = is_query_response.content.lower().strip()
        logger.info(f"Classificator: is a query, {is_query}")

        if any([is_query.startswith(x) for x in ["yes", "ja"]]):
            return "retrieve"
        else:
            return "call_model"

    async def retrieve(state: State) -> State:
        state["docs"] += await history_aware_retriever.ainvoke({"input": state["messages"][-1].content, "chat_history": state["messages"][-6:]})
        logger.info(f"Retrieved documents: {state["docs"]}")
        return state

    async def generate(state: State) -> State:
        state["answer"] = await rag_chain.ainvoke({"context": state["docs"], "question": state["messages"][-1]})
        logger.info(f"Generated answer {state["answer"]}")
        return state

    async def grade_answer(state: State) -> Literal["save_answer", "call_model"]:
        grade = await llm.ainvoke(f"I give you a text of answer. Your task is to decide if the text means \"I don't know\". Output must be just yes or no.\nText: {state["answer"]}")
        grade_content = grade.content.lower().strip()
        logger.info(f"Grading answer, {grade_content}")

        if any([grade_content.startswith(x) for x in ["no", "nein"]]):
            return "save_answer"
        else:
            return "call_model"

    async def call_model(state: State) -> State:
        msg = ""
        async for chunk in llm.astream([SystemMessage(content="Always return output in German."),
                                        state["messages"][-1]]):
            print(chunk.content, end="")
            msg += chunk.content
        state["messages"].append(AIMessage(content=msg))
        logger.info(f"Calling model, {state["messages"][-1]}")
        return state

    async def save_information(state: State) -> State:
        logger.info("Save information")
        text, embs, metadata = await process_chunk(state["messages"][-1].content.replace("#save", ""),
                                                   "chat_message",
                                                   datetime.now())
        text_emb_data = [(text, emb) for emb in embs if emb]
        logger.info("Adding embeddings")
        db.add_embeddings(text_emb_data,
                          metadatas=[metadata] * len(embs),
                          index_name=index_name
                          )
        state["messages"].append(AIMessage(f"Informationen werden im Index {index_name} gespeichert"))
        print(state["messages"][-1].content)
        return state

    async def save_answer(state: State) -> State:
        logger.info("Save answer")
        state["messages"].append(AIMessage(state["answer"]))
        print(state["messages"][-1].content)
        return state

    if verbose:
        logger.addHandler(logging.StreamHandler())

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

    embedder = OllamaEmbeddings(model="llama3")
    db_kwargs = json.load(open("creds.json", "rb"))["OPEN_SEARCH_KWARGS"]
    db_kwargs["http_auth"] = (db_kwargs.get("http_auth", {}).get("login"), db_kwargs.get("http_auth", {}).get("password"))
    index_name = db_kwargs.get("index_name")
    db = OpenSearchVectorSearch(
        embedding_function=embedder,
        **db_kwargs
    )

    llm = ChatOllama(model=model, temperature=0)
    retriever = db.as_retriever()

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_chain = rag_prompt | add_lang_prompt | llm | StrOutputParser()

    workflow = StateGraph(State)
    workflow.add_node("set_initial_state", set_initial_state)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("call_model", call_model)
    workflow.add_node("save_information", save_information)
    workflow.add_node("save_answer", save_answer)

    workflow.add_conditional_edges(START, has_save_tag)
    workflow.add_conditional_edges("set_initial_state", cls_query)
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges("generate", grade_answer)
    workflow.add_edge("call_model", END)
    workflow.add_edge("save_information", END)
    workflow.add_edge("save_answer", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}
    return graph, config

async def main_gr():
    import gradio as gr

    async def chat_response(input_message, chat_history):
        start = datetime.now()
        logger.info(f"Input message: {input_message}")
        output = await agent.ainvoke({"messages": [input_message]}, config)
        chat_history.append((input_message, output["messages"][-1].content))
        logger.info(f"Response time: {datetime.now() - start}")

        return "", chat_history

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", default="llama3", help="Model to use")
    #
    # args = parser.parse_args()

    agent, config =  await setup_agent(model="llama3", verbose=False)
    logger.info("Agent started")

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="tuples")
        message = gr.Textbox(placeholder="Type a message...", label="Your Message")
        submit = gr.Button("Send")

        # Bind the async function to the submit button
        submit.click(chat_response, inputs=[message, chatbot], outputs=[message, chatbot])

    demo.launch(share=True)

if __name__ == "__main__":
    asyncio.run(main_gr())