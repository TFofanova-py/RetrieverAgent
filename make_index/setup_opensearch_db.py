import json
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
import ollama
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from datetime import datetime
import asyncio
from typing import List, Tuple, Literal
import csv

logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/indexing.log")]
    )
logger = logging.getLogger(__name__)

chunk_auditor_prompt = """
Your task is to write up to 2 questions to the text below which an ISMS (information security management system) auditor can asks and DO REQUIRE a respondent to get you a practical example. These questions shouldn't concern budget or money.
Language of output is German.
Output must be a list without any additional thoughts and explanations.
Example:
1. Wie werden allgemeine Informationssicherheitsziele formuliert und welchen Einfluss haben diese auf den Sicherheitsprozess?
Nenne mir mehrere Beispiele, wie man Informationssicherheitsziele formulieren kann. Nenne mir mehrere Bespiele, welchen Einfluss diese Ziele auf den Sicherheitsprozess haben können.
2. Welche Rahmenbedingungen werden für den Sicherheitsprozess ermittelt und wie wird dieser Prozess gestaltet, wenn es beispielsweise eine Veränderung in der Organisation gibt? Nenne mir ein praktisches Beispiel.
"""

define_topic_prompt = """
Your task is to classify text for one of topics: {topics}. 
Return only one topic name without any thoughts and explanations.
"""

TOPICS = [
    "Context of the Organization",
    "Leadership and Commitment",
    "IS Objectives",
    "IS Policy",
    "Roles, Responsibilities and Competencies",
    "Risk Management",
    "Performance/Risk/Compliance Monitoring",
    "Documentation",
    "Communication",
    "Awareness",
    "Supplier Relationships",
    "Internal Audit",
    "Incident Management",
    "Continual Improvement"
]



def get_chunks(paths: List[Tuple[str, datetime, int]], chunk_size:int = 1000, chunk_overlap:int = 50):
    for i, (path, dt, start_page) in enumerate(paths):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[max(start_page - 1, 0):]:
                text += page.extract_text() + "\n"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n"]
        )

        yield path, dt, text_splitter.split_text(text)

async def define_topic(text: str, prompt: str = define_topic_prompt, model: str = "llama3") -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt.replace("{topics}", str(TOPICS))},
            {"role": "user", "content": text}]
    )
    return response["message"]["content"]


async def make_questions(chunk, model: str = "llama3", prompt: str = chunk_auditor_prompt):

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk}]
    )
    response_content = response["message"]["content"]
    questions = [re.sub(r"[\d]+\.", "", x).strip()
                          for x in re.findall(r"[\d]+\..+", response_content)]
    questions = [x for x in questions if x]
    topic = await define_topic(chunk)
    if not questions:
        print(response_content)
    return {"text": chunk,
            "questions": questions,
            "topic": re.sub(r"^[\'\"](.+)[\'\"]$", "\\1", topic)
            }


async def process_chunk(chunk: str, source: str, dt: datetime, model: str = "llama3.1", mode: Literal["text", "questions", "both"]= "text", out_file: str = None):
    if mode == "questions":  # to make index based on embedded questions
        doc = await make_questions(chunk=chunk, prompt=chunk_auditor_prompt)
        # input_list = doc.get("questions", [])
        with open(out_file, "a") as out:
            out.write("\n".join(doc.get("questions")))
        return None

    elif mode == "both":  # to generate questions for dataset and embedding text and questions for index
        assert out_file is not None, "you must provide an output file for questions"
        doc = await make_questions(chunk=chunk)
        with open(out_file, "a", newline="") as f_out:
            csv_writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
            rows = [[doc.get("topic"), x] for x in doc.get("questions", [])]
            csv_writer.writerows(rows)
        texts = doc.get("questions")
        metadata = [{"source": source, "topic": doc.get("topic"), "timestamp": dt, "type": "question"}] * len(texts)
        emb_response = ollama.embed(model=model, input=texts)
        input_list = [chunk]
        texts.append(chunk)
        metadata.append({"source": source, "topic": doc.get("topic"), "timestamp": dt, "type": "answer"})
        emb_response["embeddings"].extend(ollama.embed(model=model, input=input_list).get("embeddings"))
        return texts, emb_response["embeddings"], metadata
    else:  # to make index based on embedded texts
        input_list = [chunk]
        metadata = {"source": source, "timestamp": dt}
        emb_response = ollama.embed(model=model, input=input_list)
        return [chunk], emb_response["embeddings"], metadata


async def main(verbose: bool = True):
    if verbose:
        logger.addHandler(logging.StreamHandler())
    embedder = OllamaEmbeddings(model="llama3.1")

    kwargs = json.load(open("creds.json", "rb"))["OPEN_SEARCH_KWARGS"]
    kwargs["http_auth"] = (kwargs.get("http_auth", {}).get("login"), kwargs.get("http_auth", {}).get("password"))
    index_name = kwargs.get("index_name")

    db = OpenSearchVectorSearch(embedding_function=embedder, **kwargs)
    if db.index_exists(kwargs.get(index_name)):
        confirmation = input("Do you want to DELETE existing index? [Yes/No] >> ")
        if confirmation.lower().strip().startswith("yes"):
            db.delete_index(index_name)

    start = datetime.now()
    paths = [
        # ("Docs/240722_nis2-regierungsentwurf.pdf", datetime(year=2024, month=7, day=22)),
        # ("Docs/CELEX_02022L2555-20221227_DE_TXT.pdf", datetime(year=2022, month=12, day=27)),
        # ("Docs/Leitfaden_zur_Basis-Absicherung.pdf", datetime(year=2020, month=11, day=1)),
        # ("Docs/standard_200_1.pdf", datetime(year=2020, month=11, day=1)),
        ("Docs/standard_200_2.pdf", datetime(year=2020, month=11, day=2), 7),
        ("Docs/it-sicherheitsgesetz.pdf", datetime(year=2020, month=11, day=1), 1),
        ("Docs/ISACA Implementierungsleitfaden ISMS 2022.pdf", datetime(2022, 1, 1), 9),
    ]

    for source, dt, chunks in get_chunks(paths=paths, chunk_size=1500):
        logger.info(f"Splitting {source} for chunks, {datetime.now() - start}, OK")

        for i, chunk in enumerate(chunks):

            start = datetime.now()
            response = await process_chunk(chunk, source, dt, mode="both", out_file="audit_questions.txt")
            if response:
                texts, embs, metadatas = response

                embs = [x for x in embs if x]

                if not embs:
                    continue

                if db.index_exists():
                    text_emb_data = [(text, emb) for text, emb in zip(texts, embs) if emb]
                    db.add_embeddings(text_emb_data,
                                      metadatas=metadatas,
                                      index_name=index_name
                                      )
                else:
                    db = OpenSearchVectorSearch.from_embeddings(
                        embs,
                        texts,
                        embedder,
                        metadatas=metadatas,
                        **kwargs
                    )
                logger.info(f"Indexing {source}, chunk {i} / {len(chunks)}, {datetime.now() - start}, OK")
    logger.info("Success")


if __name__ == '__main__':
    asyncio.run(main())


# docker run -it -p 9200:9200 -p 9600:9600 -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=piskarev_RAG_24 -e "discovery.type=single-node" --name opensearch-node -d opensearchproject/opensearch:latest
# ollama run llama3

