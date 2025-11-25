from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import json
from pathlib import Path
import os

DOCS_JSONL = Path(__file__).resolve().parents[1] / "data" / "docs" / "movie_docs.jsonl"
PERSIST_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_store"

def build_vectorstore(openai_api_key=None):
    os.environ["OPENAI_API_KEY"] = openai_api_key or os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    docs = []
    with open(DOCS_JSONL, "r", encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(Document(page_content=d["text"], metadata=d["meta"]))
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=str(PERSIST_DIR))
    vectordb.persist()
    print("Chroma vectorstore built at", PERSIST_DIR)
    return vectordb

if __name__ == "__main__":
    build_vectorstore()
