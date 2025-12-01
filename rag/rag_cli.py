# rag/rag_cli.py
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent   # .../rackhostllm
DB_DIR   = BASE_DIR / "chroma_kb"

client = chromadb.PersistentClient(path="../chroma_kb")
collection = client.get_collection("rackhost_kb")


embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve(query, top_k=5):
    q_emb = embedder.encode([query])[0].tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
    )
    return res

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "cPanel bejelentkezés"
    res = retrieve(query)
    for i in range(len(res["ids"][0])):
        print("---")
        print("TITLE:", res["metadatas"][0][i].get("title"))
        print("URL:  ", res["metadatas"][0][i].get("url"))
        print("TEXT:", res["documents"][0][i][:400], "...")

from answer_engine import synthesize_answer

contexts = [m["document"] for m in results]
answer = synthesize_answer(query, contexts)
print("\n\n=== FINAL ANSWER ===\n")
print(answer)
