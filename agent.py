import asyncio
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from ollama import AsyncClient
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
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# _set_env("TAVILY_API_KEY")
# web_search_tool = TavilySearchResults(k=3)

logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler("agent.log")]
    )
logger = logging.getLogger(__name__)


class State(MessagesState):
    is_first_attempt: bool
    docs: List[Document]
    answer: str
    verbose: bool


async def setup_agent(verbose: bool = False) -> Tuple[CompiledStateGraph, dict]:
    def set_initial_state(state: State):
        state["is_first_attempt"] = True
        state["answer"] = ""
        state["verbose"] = verbose
        state["docs"] = []
        return state

    async def retrieve(state: State) -> State:
        state["docs"] += await history_aware_retriever.ainvoke({"input": state["messages"][-1].content, "chat_history": state["messages"][-6:]})
        logger.info(f"Retrieved documents: {state["docs"]}")
        return state

    async def generate(state: State) -> State:
        answer = await rag_chain.ainvoke({"context": state["docs"], "question": state["messages"][-1]})
        logger.info(f"Generated answer {answer}")
        response = await llm.ainvoke(f"Rewrite the answer in the same language as question.\nAnswer: {answer}\nQuestion: {state["messages"][-1].content}")
        state["answer"] = response.content
        logger.info(f"Rewrited answer {state['answer']}")
        return state

    async def grade_answer(state: State) -> Literal[END, "call_model", "add_context"]:
        if not state["is_first_attempt"]:
            return "call_model"

        grade = await llm.ainvoke(
            f"Decide if the answer {state["answer"]} is answering to the question {state["messages"][-1]}. Output must be just yes/no")
        grade_content = grade.content.lower().strip()
        logger.info(f"Grading answer, {grade_content}")

        if grade_content.startswith("yes"):
            state["messages"] += state["answer"]
            print(state["answer"])
            return END
        elif not state["is_first_attempt"]:
            return "call_model"
        else:
            print(f"Intermediate answer: {state['answer']}, trying to add additional context")
            return "add_context"

    async def call_model(state: State) -> State:
        state["messages"] += await llm.ainvoke([state["messages"][-1]])
        logger.info(state["messages"])
        logger.info(f"Calling model, {state["messages"][-1]}")
        return state

    async def add_context(state: State) -> State:
        state["is_first_attempt"] = False
        context = await llm.ainvoke(
            f"Add context to question {state["messages"][-1]} to improve the answer {state['answer']}. Length of the context should be 3-4 sentences. Return only context without additional details.")

        state["docs"].append(Document(page_content=context.content))
        logger.info(f"Adding context to question, {context.content}")
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
    db = OpenSearchVectorSearch(
        embedding_function=embedder,
        **db_kwargs
    )

    llm = ChatOllama(model="llama3", temperature=0)
    retriever = db.as_retriever()

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_chain = rag_prompt | llm | StrOutputParser()

    workflow = StateGraph(State)
    workflow.add_node("set_initial_state", set_initial_state)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("call_model", call_model)
    workflow.add_node("add_context", add_context)

    workflow.add_edge(START, "set_initial_state")
    workflow.add_edge("set_initial_state", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges("generate", grade_answer)
    workflow.add_edge("add_context", "generate")
    workflow.add_edge("call_model", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}
    return graph, config


if __name__ == "__main__":
    agent, config = asyncio.run(setup_agent(verbose=False))
    logger.info("Agent started")

    while True:
        input_message = input(">>")  #  "Ich arbeite in einem Krankenhaus. Muss ich die Vorschriften aus diesem Gesetz befolgen?"
        start = datetime.now()
        output = asyncio.run(agent.ainvoke({"messages": [input_message]}, config))
        logger.info(f"Response time: {datetime.now() - start}")
