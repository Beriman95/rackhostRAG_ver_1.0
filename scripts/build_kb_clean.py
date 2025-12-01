import json
import hashlib
from pathlib import Path

IN_PATH = Path("kb_export.jsonl")
OUT_PATH = Path("kb_clean.jsonl")

SOURCE_NAME = "rackhost.hu/tudasbazis"


def extract_category(url: str, fallback: str = "") -> str:
    """
    URL-ből próbál kategóriát kivenni:
    https://www.rackhost.hu/tudasbazis/<category>/slug/
    """
    if "/tudasbazis/" not in url:
        return fallback or ""
    try:
        after = url.split("/tudasbazis/", 1)[1]  # "altalanos/valami..."
        parts = after.strip("/").split("/")
        if parts:
            return parts[0]
    except Exception:
        pass
    return fallback or ""


def make_id(url: str) -> str:
    """
    Stabil, determinisztikus ID: URL MD5 hash-ből.
    """
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"kb-{h[:12]}"


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    out_f = OUT_PATH.open("w", encoding="utf-8")

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)

            url = (raw.get("url") or "").strip()
            title = (raw.get("title") or "").strip()
            html = raw.get("html") or ""
            body = (raw.get("text") or "").strip()
            raw_cat = raw.get("category") or ""

            category = extract_category(url, raw_cat)

            doc = {
                "id": make_id(url),
                "source": SOURCE_NAME,
                "url": url,
                "title": title,
                "category": category,
                "body": body,
                "html": html,
            }

            out_f.write(json.dumps(doc, ensure_ascii=False))
            out_f.write("\n")

    out_f.close()
    print(f"OK – írtam: {OUT_PATH}")


if __name__ == "__main__":
    main()
