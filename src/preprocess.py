from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.ingest import load_dataset
from pathlib import Path
import json
from tqdm import tqdm

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_documents(chunk_size=1000, chunk_overlap=200):
    df = load_dataset()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = f"Title: {row['Title']}\nYear: {row.get('Year','')}\nPlot:\n{row['text']}"
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = {
                "id": f"{row['doc_id']}_{i}",
                "text": chunk,
                "meta": {"title": row['Title'], "year": row.get('Year')}
            }
            docs.append(doc)
    # write docs to disk as jsonlines
    out = OUT_DIR / "movie_docs.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved {len(docs)} chunks to {out}")
    return out

if __name__ == "__main__":
    generate_documents()
