# rag/rag_cli.py

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent.parent    # rackhostllm
DB_DIR = BASE_DIR / "chroma_kb"                      # UGYANAZ mint build_index.py

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_collection("rackhost_kb")    # UGYANAZ a név


embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve(query: str, top_k: int = 5):
    q_emb = embedder.encode([query]).tolist()[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    docs = []
    for i in range(len(res["ids"][0])):
        docs.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i]
        })
    return docs

def build_prompt(query: str, docs):
    context_blocks = []
    for d in docs:
        meta = d["meta"]
        header = f"[{meta.get('title','')} | {meta.get('url','')} | {meta.get('category','')}]"
        context_blocks.append(header + "\n" + d["text"])
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "Te egy Rackhost tudásbázis-asszisztens vagy. "
        "Csak az alábbi KONTEKSTUS alapján válaszolj. "
        "Ha valamire nincs egyértelmű válasz, mondd ki, hogy nem egyértelmű, "
        "és utalj a legrelevánsabb cikk(ek)re (URL)."
    )

    user_prompt = f"""KONTEKSTUS:
{context}

KÉRDÉS:
{query}

VÁLASZ (röviden, lényegre törően, magyarul):"""

    return system, user_prompt

def call_ollama(system_prompt: str, user_prompt: str) -> str:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

def answer(query: str):
    docs = retrieve(query, top_k=5)
    if not docs:
        return "Nincs találat a tudásbázisban erre a kérdésre."

    system_prompt, user_prompt = build_prompt(query, docs)
    reply = call_ollama(system_prompt, user_prompt)

    debug_sources = "\n\nForrások:\n" + "\n".join(
        f"- {d['meta'].get('title','')} | {d['meta'].get('url','')}"
        for d in docs
    )

    return reply + debug_sources

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Adj meg egy kérdést argumentumként.")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print("KÉRDÉS:", query)
    print("\nVÁLASZ:\n")
    print(answer(query))

