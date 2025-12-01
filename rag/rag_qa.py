import sys
from pathlib import Path
from typing import Union  # ÚJ: A Python 3.9 kompatibilitás miatt

import torch
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================== ALAP BEÁLLÍTÁSOK ==================

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_kb"
COLLECTION_NAME = "rackhost_kb"

# Eszköz
# Az MPS/VRAM problémák miatt alapértelmezetten CPU-ra állítva a stabilitás érdekében.
# Ha nagyobb modell és több RAM áll rendelkezésre, át lehet állítani "mps"-re vagy "cuda"-ra.
if torch.cuda.is_available():
    DEVICE = "cuda"
# elif torch.backends.mps.is_available():
#     DEVICE = "mps" # Kikapcsolva 8GB RAM esetén
else:
    DEVICE = "cpu"

print(f"Eszköz (DEVICE): {DEVICE}") # Kiegészítő kiírás

# Ugyanaz az embedder, mint indexelésnél
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Kicsi seq2seq modell – flan-t5-small
LLM_MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
# A modell betöltése a kiválasztott eszközre
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(DEVICE)
model.eval()

MAX_CONTEXT_CHARS = 800
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 120  # 2–4 mondat


# ================== CHROMA ==================

def get_collection():
    client = PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    return collection


# ================== RAG LÉPÉSEK ==================

def embed_query(query: str):
    vec = embedder.encode([query])
    return vec[0].tolist()


def retrieve_best_context(question: str):
    collection = get_collection()
    q_emb = embed_query(question)

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=1,
        include=["documents", "metadatas"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    if not docs:
        return None

    txt = docs[0]
    meta = metas[0] if metas else {}

    return {
        "text": txt,
        "title": meta.get("title", ""),
        "url": meta.get("url", ""),
        "category": meta.get("category", ""),
    }


# Típus-annotáció javítva Python 3.9-re
def build_prompt(question: str, ctx: Union[dict, None]) -> str:
    if ctx is None:
        # Semmi találat → általános fallback
        return (
            "Válaszolj magyarul, röviden és érthetően 2–4 mondatban a kérdésre.\n\n"
            f"Kérdés: {question}\n\n"
            "Válasz:"
        )

    text = (ctx.get("text") or "").strip()
    text = text.replace("\r", " ").replace("\n\n", "\n")
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS]

    title = ctx.get("title") or ""
    category = ctx.get("category") or ""

    # Teljesen magyar, egyszerű QA prompt
    prompt = (
        "Olvasd el az alábbi Rackhost tudásbázis-részletet, majd válaszolj a kérdésre "
        "magyarul, 2–4 mondatban, laikus ügyfél számára is érthetően.\n\n"
    )

    if title:
        prompt += f"Cím: {title}\n"
    if category:
        prompt += f"Kategória: {category}\n"

    prompt += "\nSzöveg:\n"
    prompt += text
    prompt += "\n\nKérdés: " + question + "\n"
    prompt += "Válasz: A fenti szöveg alapján, magyarul, 2-4 mondatban, érthetően:"

    return prompt


# ================== GENERÁLÁS ==================

def generate_answer(prompt: str) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=True,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            early_stopping=False, # MÓDOSÍTVA: True helyett False, a kis modell befulladásának elkerülésére
        )

    gen_ids = out_ids[0]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


# ================== FŐ FÜGGVÉNY ==================

def fallback_snippet_answer(question: str, ctx: Union[dict, None]) -> str:
    if ctx is None:
        return "Erre a kérdésre nem találtam választ a tudásbázisban."

    text = (ctx.get("text") or "").strip()
    title = ctx.get("title") or ""
    url = ctx.get("url") or ""

    # Kicsit több szöveg, hogy ne legyen túl rövid a snippet
    snippet = " ".join(text.split()[:50]) + "..."
    if len(snippet) > 400:
        snippet = snippet[:400] + "..."

    parts = []
    if title:
        parts.append(f"[KB cikk] {title}")
    parts.append(snippet)
    if url:
        parts.append(f"(Forrás: {url})")

    return "\n".join(parts)


def answer_question(question: str):
    ctx = retrieve_best_context(question)

    prompt = build_prompt(question, ctx)
    llm_answer = generate_answer(prompt)
    
    # A kiíratás meggyőződése érdekében, hogy a prompt futott:
    # print("\n--- PROMPT ---\n", prompt, "\n--------------\n")
    # print("--- LLM VÁLASZ HOSSZA: ", len(llm_answer), "\n--------------\n")

    # Ha a modell válasza túl rövid vagy láthatóan szemét, fallback
    if not llm_answer or len(llm_answer) < 20:
        final = fallback_snippet_answer(question, ctx)
        final = "⚠️ RÖVID VÁLASZ/HIBA (FALLBACK):\n" + final
    else:
        final = "✅ LLM VÁLASZ:\n" + llm_answer

    print(final)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Használat: python3 rag/rag_qa.py "kérdés szövege"')
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    answer_question(q)