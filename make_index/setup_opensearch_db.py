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

logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler("logs/indexing.log")]
    )
logger = logging.getLogger(__name__)

chunk_prompt = """
Your task is to write 7 questions to the text below, which you can ask the ISMS (information security management system) assistant. Ask questions from the perspective of a company that needs to apply this law in its daily operations.
Language of output is German.
Output must be a list of questions without any additional thoughts and explanations.
Examples:
1. Was bedeutet NIS2 für mein Unternehmen?
2. Welche Produkte und Services muss ich einsetzen, um NIS2 gerecht zu werden?
3. Welche Prozesse müssen gemäß NIS2 in meinem Unternehmen vorhanden sein?
4. Welche Auswirkungen hat NIS2 auf mich, als einen Mitarbeiter der IT Abteilung?
5. Welche Auswirkungen hat NIS2 auf das Unternehmen, in dem ich beschäftigt bin?
"""


def get_chunks(paths: List[Tuple[str, datetime]], chunk_size:int = 1000, chunk_overlap:int = 50):
    for i, (path, dt) in enumerate(paths):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n"]
        )
        start_chunk = 0 if i == 0 else 0

        yield path, dt, text_splitter.split_text(text)[start_chunk:]


async def make_questions(chunk, model: str = "llama3", safe:bool = True):

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": chunk_prompt},
            {"role": "user", "content": chunk}]
    )
    response_content = response["message"]["content"]
    questions = [re.sub(r"[\d]+\.", "", x).strip()
                          for x in re.findall(r"[\d]+\..+", response_content)]
    questions = [x for x in questions if x]
    if not questions:
        print(response_content)
    if safe and questions:
        with open("questions.txt", "a") as f_out:
            f_out.write("\n".join(questions) + "\n")
    return {"text": chunk,
            "questions": questions
            }


async def process_chunk(chunk: str, source: str, dt: datetime, model: str = "llama3", mode: Literal["text", "questions"]= "questions"):
    if mode == "questions":
        doc = await make_questions(chunk=chunk)
        input_list = doc.get("questions", [])
    else:
        input_list = [chunk]
    metadata = {"source": source, "timestamp": dt}
    emb_response = ollama.embed(model="llama3.1", input=input_list)
    return chunk, emb_response["embeddings"], metadata


async def main(verbose: bool = True):
    if verbose:
        logger.addHandler(logging.StreamHandler())
    embedder = OllamaEmbeddings(model="llama3")

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
        ("Docs/240722_nis2-regierungsentwurf.pdf", datetime(year=2024, month=7, day=22)),
        ("Docs/CELEX_02022L2555-20221227_DE_TXT.pdf", datetime(year=2022, month=12, day=27)),
        ("Docs/Leitfaden_zur_Basis-Absicherung.pdf", datetime(year=2020, month=11, day=1)),
        ("Docs/standard_200_1.pdf", datetime(year=2020, month=11, day=1)),
        ("Docs/standard_200_2.pdf", datetime(year=2020, month=11, day=2)),
        ("Docs/it-sicherheitsgesetz.pdf", datetime(year=2020, month=11, day=1)),
    ]

    for source, dt, chunks in get_chunks(paths=paths, chunk_size=700):
        logger.info(f"Splitting {source} for chunks, {datetime.now() - start}, OK")

        for i, chunk in enumerate(chunks):
            start = datetime.now()
            response = await process_chunk(chunk, source, dt, mode="text")
            if response:
                text, embs, metadata = response

                embs = [x for x in embs if x]

                if not embs:
                    continue

                if db.index_exists():
                    text_emb_data = [(text, emb) for emb in embs if emb]
                    db.add_embeddings(text_emb_data,
                                      metadatas=[metadata] * len(embs),
                                      index_name=index_name
                                      )
                else:
                    text_data = [text] * len(embs)
                    db = OpenSearchVectorSearch.from_embeddings(
                        embs,
                        text_data,
                        embedder,
                        metadatas=[{"source": source}] * len(embs),
                        **kwargs
                    )
                logger.info(f"Indexing {source}, chunk {i} / {len(chunks)}, {datetime.now() - start}, OK")
    logger.info("Success")


if __name__ == '__main__':
    asyncio.run(main())


# docker run -it -p 9200:9200 -p 9600:9600 -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=piskarev_RAG_24 -e "discovery.type=single-node" --name opensearch-node -d opensearchproject/opensearch:latest
# ollama run llama3
