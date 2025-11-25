# src/utils.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

def ensure_dirs():
    for d in ["data/docs", "data/chroma_store", "logs"]:
        Path(BASE_DIR / d).mkdir(parents=True, exist_ok=True)

def load_env(env_path: str = None):
    # load .env if exists
    load_dotenv(env_path or BASE_DIR.parent / ".env")
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PERSIST_DIR": str(BASE_DIR / "data" / "chroma_store"),
        "DOCS_JSONL": str(BASE_DIR / "data" / "docs" / "movie_docs.jsonl"),
        "RAW_CSV": str(BASE_DIR / "data" / "wikipedia_movie_plots.csv"),
        "ARCHITECTURE_IMAGE": "/mnt/data/D596CC49-266A-4E6E-B326-920E986C81C2.jpeg"
    }
