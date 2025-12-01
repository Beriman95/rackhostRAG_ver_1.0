from pathlib import Path
import json
import hashlib

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent.parent        # .../rackhostllm
DB_DIR   = BASE_DIR / "chroma_kb"                        # lokális Chroma DB

KB_PATH  = BASE_DIR / "data" / "kb_chunks.jsonl"         # RAG input


def make_doc_id(obj, idx: int) -> str:
  """
  Stabil, de laza ID generálás:
  - ha van obj["id"], azt használja
  - ha nincs, url+idx-ből hash
  """
  raw = obj.get("id")
  if not raw:
    base = (obj.get("url") or "kb") + f"#{idx}"
    raw = "kb-" + hashlib.md5(base.encode("utf-8")).hexdigest()[:16]
  return str(raw)


def build_index():
  print(f"Loading from: {KB_PATH}")

  client = chromadb.PersistentClient(path=str(DB_DIR))

  collection = client.get_or_create_collection(
    name="rackhost_kb"  # fontos: ezt használja a rag_cli is
  )

  texts = []
  metadatas = []
  ids = []

  with KB_PATH.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)

      doc_id = make_doc_id(obj, i)
      body   = obj.get("body") or obj.get("text") or ""
      if not body.strip():
        continue  # üres chunk nem kell

      ids.append(doc_id)
      texts.append(body)
      metadatas.append({
        "url": obj.get("url", ""),
        "title": obj.get("title", ""),
        "category": obj.get("category", ""),
      })

  if not texts:
    print("Nincs indexelhető chunk (texts üres).")
    return

  embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
  )

  print(f"Embedding documents: {len(texts)} db")
  embeddings = embedding_fn(texts)

  collection.add(
    ids=ids,
    documents=texts,
    metadatas=metadatas,
    embeddings=embeddings,
  )

  print(f"Index építve. Összes elem: {len(ids)}")


if __name__ == "__main__":
  build_index()
