import os
from pathlib import Path
from textwrap import dedent

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

# --- KONFIG ---
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_kb"
COLLECTION_NAME = "rackhost_kb"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"  # vagy amit használsz


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(allow_reset=False)
)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)


def embed(text: str):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return resp.data[0].embedding


def retrieve(query: str, k: int = 6):
    q_emb = embed(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )
    # Chroma visszaad: ids, documents, metadatas
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return list(zip(docs, metas))


def build_context_snippet(results):
    """Források kontextusba rendezése, hogy a modell lássa,
    melyik szöveg honnan jött."""
    parts = []
    for i, (text, meta) in enumerate(results):
        title = meta.get("title") or ""
        url = meta.get("url") or ""
        parts.append(
            f"[{i+1}] TITLE: {title}\nURL: {url}\nCONTENT:\n{text}\n"
        )
    return "\n\n-----\n\n".join(parts)


def answer(query: str):
    results = retrieve(query, k=6)
    if not results:
        system = "Te egy Rackhost ügyfélszolgálati asszisztens vagy. Ha nincs adatod, mondd ki egyenesen."
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
    else:
        context = build_context_snippet(results)

        system = dedent("""
        Te egy Rackhost ügyfélszolgálati asszisztens vagy.
        Csak az alábbi KONTEKSTUS alapján válaszolj.
        Ha a kontextus nem tartalmaz választ, mondd ki egyenesen, hogy a tudásbázisban nincs rá egyértelmű adat.
        Mindig hivatkozz a releváns forrás(ok) számára (pl. [1], [3]) és ha van, említsd a URL-t.
        Ne találj ki új szabályokat vagy árakat.
        """).strip()

        user_prompt = f"""
        KÉRDÉS:
        {query}

        KONTEKSTUS (tudásbázis cikk chunkok):

        {context}
        """.strip()

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=msgs,
        temperature=0.1,
    )
    return resp.choices[0].message.content


def main():
    print("Rackhost KB RAG agent. Kilépéshez: üres sor vagy Ctrl+C.\n")
    while True:
        try:
            q = input("Kérdés: ").strip()
            if not q:
                break
            ans = answer(q)
            print("\nVálasz:\n")
            print(ans)
            print("\n" + "=" * 60 + "\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
