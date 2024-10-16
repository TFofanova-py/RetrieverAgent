import json
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from ollama import AsyncClient
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from datetime import datetime
import asyncio

logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler("indexing.log")]
    )
logger = logging.getLogger(__name__)

chunk_prompt = """
Your task is to generate about 7 questions on which follow text answers.
Generate in the same language as text.
Output must be a list of questions without any additional thoughts and explanations.
"""


def get_chunks():
    path = "Docs/240722_nis2-regierungsentwurf.pdf"
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n"]
    )

    return path, text_splitter.split_text(text)


async def make_questions(chunk, model: str = "llama3"):

    response = await AsyncClient().chat(
        model=model,
        messages=[
            {"role": "system", "content": chunk_prompt},
            {"role": "user", "content": chunk}]
    )
    response_content = response["message"]["content"]
    return {"text": chunk,
            "questions": [re.sub(r"[\d]+\.", "", x).strip()
                          for x in re.findall(r"[\d]+\..+", response_content)],
            }


async def process_chunk(chunk: str, source: str, model: str = "llama3"):
    doc = await make_questions(chunk=chunk)
    text = doc.get("text")
    metadata = {"source": source}
    chunk_embeddings = OllamaEmbeddings(model="llama3").embed_documents(doc.get("questions", []))
    return text, chunk_embeddings, metadata


async def main(verbose: bool = True):
    if verbose:
        logger.addHandler(logging.StreamHandler())
    embedder = OllamaEmbeddings(model="llama3")

    kwargs = json.load(open("creds.json", "rb"))["OPEN_SEARCH_KWARGS"]
    kwargs["http_auth"] = (kwargs.get("http_auth", {}).get("login"), kwargs.get("http_auth", {}).get("password"))
    index_name = kwargs.get("index_name")

    db = OpenSearchVectorSearch(embedding_function=embedder, **kwargs)
    if db.index_exists(kwargs.get(index_name)):
        db.delete_index(index_name)

    start = datetime.now()
    source, chunks = get_chunks()
    logger.info(f"Splitting {source} for chunks, {datetime.now() - start}, OK")

    i_start = 0  # 610 total
    tasks = []
    for i, chunk in enumerate(chunks[i_start:]):
        tasks.append(process_chunk(chunk, source))
    results = await asyncio.gather(*tasks)

    for text, embs, metadata in results:
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
        logger.info(f"Indexing {source} for chunk {i + i_start}, OK")
    logger.info("Success")


if __name__ == '__main__':
    asyncio.run(main())


# docker run -it -p 9200:9200 -p 9600:9600 -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=piskarev_RAG_24 -e "discovery.type=single-node" --name opensearch-node -d opensearchproject/opensearch:latest
# ollama run llama3