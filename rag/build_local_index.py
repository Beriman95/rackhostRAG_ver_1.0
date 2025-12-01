# scripts/build_index.py

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import json

BASE_DIR = Path(__file__).resolve().parent.parent  # rackhostllm gyökér
DB_DIR = BASE_DIR / "chroma_kb"                    # IDE épül az index

def build_index():
    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name="rackhost_kb"  # <<< EZ A NÉV KÖTELEZŐEN UGYANAZ, mint a rag_cli-ben
    )

    # például:
    kb_path = BASE_DIR / "data" / "kb_chunks.jsonl"
    print(f"Loading from: {kb_path}")

    texts = []
    metadatas = []
    ids = []

    with kb_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            texts.append(obj["body"])
            metadatas.append({
                "url": obj.get("url", ""),
                "title": obj.get("title", ""),
                "category": obj.get("category", "")
            })

    # SentenceTransformers embedder – lokális, ingyen
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embedding_fn(texts)
    )

    print(f"Index építve. Összes elem: {len(ids)}")

if __name__ == "__main__":
    build_index()
