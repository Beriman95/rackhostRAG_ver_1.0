#!/usr/bin/env python3
import json
import os

# ---- KONFIG ----

INPUT_PATH = "kb_clean.jsonl"     # a mostani exportod
OUTPUT_PATH = "kb_chunks.jsonl"   # ide írjuk a chunkokat

# Chunkolás karakter alapon (RAG-hoz bőven jó első körnek)
MAX_CHARS = 1200      # egy chunk max hossza
OVERLAP_CHARS = 200   # átfedés két chunk között


def chunk_text(text, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
  """
  Szöveg felvágása max_chars méretű chunkokra.
  Egyszerű karakter-alapú vágás, próbál mondathatár / szóköz felé igazodni.
  Vissza: list[str]
  """
  text = text.strip()
  if not text:
    return []

  chunks = []
  n = len(text)
  start = 0

  while start < n:
    end = min(start + max_chars, n)
    # próbáljunk szóköznél vágni, ne fél szónál
    if end < n:
      # próbáljunk visszafelé keresni egy szóközt
      cut = text.rfind(" ", start + int(max_chars * 0.6), end)
      if cut == -1:
        cut = end
      else:
        end = cut

    chunk = text[start:end].strip()
    if chunk:
      chunks.append(chunk)

    if end >= n:
      break

    # overlap-pel lépjünk tovább
    start = max(0, end - overlap)

  return chunks


def main():
  if not os.path.exists(INPUT_PATH):
    print(f"HIBA: Nem találom az input fájlt: {INPUT_PATH}")
    return

  total_articles = 0
  total_chunks = 0

  with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
       open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in fin:
      line = line.strip()
      if not line:
        continue

      try:
        article = json.loads(line)
      except json.JSONDecodeError as e:
        print(f"JSON hiba, sor kihagyva: {e}")
        continue

      doc_id = article.get("id") or ""
      body = article.get("body") or ""
      url = article.get("url") or ""
      title = article.get("title") or ""
      category = article.get("category") or ""

      if not doc_id or not body.strip():
        continue

      total_articles += 1

      chunks = chunk_text(body)
      if not chunks:
        continue

      for i, chunk in enumerate(chunks):
        chunk_obj = {
          "doc_id": doc_id,
          "chunk_local_index": i,
          "chunk_id": f"{doc_id}-chunk-{i}",
          "source": article.get("source") or "rackhost.hu/tudasbazis",
          "url": url,
          "title": title,
          "category": category,
          "text": chunk,
        }
        fout.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
        total_chunks += 1

  print(f"Kész. Cikkek: {total_articles}, chunkok: {total_chunks}")
  print(f"Kimenet: {OUTPUT_PATH}")


if __name__ == "__main__":
  main()
