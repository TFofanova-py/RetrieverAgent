from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate
import json


embedder = OllamaEmbeddings(model="llama3")

db_kwargs = json.load(open("../creds.json", "rb"))["OPEN_SEARCH_KWARGS"]
db_kwargs["http_auth"] = (db_kwargs.get("http_auth", {}).get("login"), db_kwargs.get("http_auth", {}).get("password"))
db = OpenSearchVectorSearch(
    embedding_function=embedder,
    **db_kwargs
)

llm = OllamaLLM(model="llama3")
retriever = db.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. Answer must be in the same language as the question"
    "\n\n"
    "{context}"
)

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
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

if __name__ == "__main__":
    chat_history = []

    for _ in range(5):
        query = input()
        response = rag_chain.invoke({"input": query, "chat_history": chat_history[:6]})
        chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=response["answer"]),
            ]
        )
