import sys
from pathlib import Path
from typing import Union
import requests
import json

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ================== ALAP BEÁLLÍTÁSOK ==================

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_kb"
COLLECTION_NAME = "rackhost_kb"

# Embedder - ugyanaz marad
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Ollama beállítások
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"

MAX_CONTEXT_CHARS = 1200
TOP_K_DOCS = 2

# ================== CHROMA ==================

def get_collection():
    client = PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    return collection

# ================== RAG LÉPÉSEK ==================

def embed_query(query: str):
    vec = embedder.encode([query])
    return vec[0].tolist()

def retrieve_best_contexts(question: str, top_k: int = TOP_K_DOCS):
    collection = get_collection()
    q_emb = embed_query(question)

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    if not docs:
        return []

    contexts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        if dist > 1.5:
            continue
        
        contexts.append({
            "text": doc,
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "category": meta.get("category", ""),
            "distance": dist,
            "rank": i + 1
        })
    
    return contexts

def build_prompt(question: str, contexts: list) -> str:
    if not contexts:
        return (
            "Válaszolj magyarul, szakszerűen, de érthetően 2-4 mondatban.\n\n"
            f"Kérdés: {question}\n\n"
            "Válasz:"
        )

    context_parts = []
    for i, ctx in enumerate(contexts[:2], 1):
        text = ctx.get("text", "").strip()
        text = text.replace("\r", " ").replace("\n\n", "\n")
        
        if len(text) > MAX_CONTEXT_CHARS // len(contexts):
            text = text[:MAX_CONTEXT_CHARS // len(contexts)]
        
        title = ctx.get("title", "")
        if title:
            context_parts.append(f"[Dokumentum {i}: {title}]")
        else:
            context_parts.append(f"[Dokumentum {i}]")
        
        context_parts.append(text)
        context_parts.append("")

    combined_context = "\n".join(context_parts)

    prompt = f"""Te egy Rackhost ügyfélszolgálati munkatárs vagy. Az alábbiakban tudásbázis részleteket kapsz, majd egy ügyfél kérdését.

TUDÁSBÁZIS:
{combined_context}

FELADAT:
- Válaszolj a kérdésre MAGYARUL, 2-4 mondatban
- Használd a fenti dokumentumokat
- Légy pontos, de közérthető
- Ha a dokumentumok nem tartalmaznak releváns infót, mondd el őszintén

ÜGYFÉL KÉRDÉSE: {question}

VÁLASZ:"""

    return prompt

# ================== OLLAMA GENERÁLÁS ==================

def generate_with_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Union[str, None]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 200,
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        answer = result.get("response", "").strip()
        
        return answer if answer else None
        
    except requests.exceptions.ConnectionError:
        print("❌ HIBA: Nem lehet csatlakozni az Ollama-hoz!")
        print("   Futtasd: brew services start ollama")
        return None
    except Exception as e:
        print(f"❌ Ollama hiba: {e}")
        return None

# ================== FALLBACK ==================

def fallback_snippet_answer(contexts: list) -> str:
    if not contexts:
        return "Erre a kérdésre nem találtam választ a tudásbázisban."

    ctx = contexts[0]
    text = ctx.get("text", "").strip()
    title = ctx.get("title", "")
    url = ctx.get("url", "")

    snippet = " ".join(text.split()[:60]) + "..."
    if len(snippet) > 500:
        snippet = snippet[:500] + "..."

    parts = []
    if title:
        parts.append(f"📄 {title}")
    parts.append(snippet)
    if url:
        parts.append(f"\n🔗 Forrás: {url}")

    return "\n".join(parts)

# ================== FŐ FÜGGVÉNY ==================

def answer_question(question: str):
    print(f"\n🔍 Keresés a tudásbázisban: '{question}'\n")
    
    contexts = retrieve_best_contexts(question, top_k=TOP_K_DOCS)
    
    if not contexts:
        print("⚠️  Nem találtam releváns dokumentumot.\n")
        return
    
    print(f"✓ {len(contexts)} releváns dokumentum találva")
    for ctx in contexts:
        print(f"  • {ctx.get('title', 'N/A')} (távolság: {ctx.get('distance', 0):.3f})")
    print()
    
    prompt = build_prompt(question, contexts)
    
    print("🤖 LLM válasz generálása...\n")
    llm_answer = generate_with_ollama(prompt)
    
    if not llm_answer or len(llm_answer) < 30:
        print("⚠️  FALLBACK MÓD (LLM hiba):\n")
        final = fallback_snippet_answer(contexts)
    else:
        final = llm_answer
    
    print("=" * 70)
    print("VÁLASZ:")
    print("=" * 70)
    print(final)
    print("=" * 70)
    
    if contexts:
        print("\n📚 Források:")
        for ctx in contexts[:2]:
            if ctx.get("url"):
                print(f"  • {ctx.get('url')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Használat: python3 rag_qa_ollama.py "kérdés szövege"')
        print(f'\nJelenleg használt modell: {OLLAMA_MODEL}')
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    answer_question(q)
