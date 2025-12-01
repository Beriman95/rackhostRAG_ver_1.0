import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.rackhost.hu"
KB_ROOT = "https://www.rackhost.hu/tudasbazis/"

session = requests.Session()
session.headers.update({
    "User-Agent": "Rackhost-KB-Export/1.0 (internal)"
})

def get_soup(url):
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def collect_category_urls():
    soup = get_soup(KB_ROOT)
    cats = set()

    # TODO: pontos selektor finomhangolása a konkrét HTML alapján
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/tudasbazis/" in href and href.rstrip("/").count("/") >= 4:
            # pl. /tudasbazis/tarhely/ → kategória
            cats.add(urljoin(BASE_URL, href))
    return sorted(cats)

def collect_article_urls_from_category(cat_url):
    urls = set()
    next_url = cat_url

    while next_url:
        soup = get_soup(next_url)

        # TODO: cikklista linkjei (pl. article linkek)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/tudasbazis/" in href and href.rstrip("/").count("/") > 4:
                # pl. /tudasbazis/tarhely/valami-cikk/
                urls.add(urljoin(BASE_URL, href))

        # pagináció keresése (ha van "Következő" gomb)
        next_link = soup.find("a", string=lambda s: s and "Következő" in s)
        if next_link and next_link.get("href"):
            next_url = urljoin(BASE_URL, next_link["href"])
        else:
            next_url = None

        time.sleep(0.5)

    return sorted(urls)

def parse_article(url):
    soup = get_soup(url)

    # Ezeket a selektorokat a konkrét HTML alapján kell pontosítani
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else url

    # pl. fő tartalom egy <article> vagy .single-content div
    content_el = soup.find("article")
    if not content_el:
        content_el = soup.find("div", class_="single-content") or soup

    # nyers HTML + sima text
    content_html = str(content_el)
    content_text = content_el.get_text("\n", strip=True)

    # kategória – pl. breadcrumb alapján
    cat = None
    breadcrumb = soup.find("nav", class_="breadcrumb")
    if breadcrumb:
        links = breadcrumb.find_all("a")
        if len(links) >= 2:
            cat = links[-2].get_text(strip=True)

    return {
        "url": url,
        "title": title,
        "category": cat,
        "html": content_html,
        "text": content_text
    }

def main():
    all_article_urls = set()

    category_urls = collect_category_urls()
    print("Kategóriák:", len(category_urls))

    for cat in category_urls:
        print("Kategória:", cat)
        urls = collect_article_urls_from_category(cat)
        print("  Cikkek:", len(urls))
        all_article_urls.update(urls)

    print("Összes egyedi cikk:", len(all_article_urls))

    with open("kb_export.jsonl", "w", encoding="utf-8") as f:
        for url in sorted(all_article_urls):
            try:
                art = parse_article(url)
                f.write(json.dumps(art, ensure_ascii=False) + "\n")
                time.sleep(0.5)
            except Exception as e:
                print("Hiba:", url, e)

if __name__ == "__main__":
    main()
