#!/usr/bin/env python3
"""One-time script: fetch PubMed abstracts + Wikipedia articles, embed with
BAAI/bge-small-en-v1.5, store in ChromaDB.

Run from project root:
    python scripts/build_corpus.py

Resumable: documents already in ChromaDB are skipped.
"""

import html
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen, Request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
CORPUS_RAW_DIR = Path("corpus/raw")
COLLECTION_NAME = "fitness_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

PUBMED_QUERIES = [
    "muscle hypertrophy resistance training",
    "progressive overload strength training",
    "training volume muscle growth",
    "deload recovery resistance training",
    "training frequency hypertrophy",
    "rep range strength hypertrophy",
    "muscle recovery between workouts",
    "DOMS delayed onset muscle soreness exercise",
]

WIKIPEDIA_ARTICLES = [
    "Progressive_overload",
    "Muscle_hypertrophy",
    "Delayed_onset_muscle_soreness",
    "Periodization",
    "One-repetition_maximum",
    "Strength_training",
    "Overtraining",
]

WIKIPEDIA_SECTION_ARTICLES = ["Muscle_hypertrophy", "Strength_training"]

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
WIKI_BASE = "https://en.wikipedia.org/api/rest_v1/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[\s_-]+", "_", s).strip("_")[:50]


def _clean_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_USER_AGENT = "FitNotesCoach/2.0 (fitness research corpus builder; educational project)"


def _fetch_url(url: str) -> bytes | None:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(req, timeout=15) as resp:
            return resp.read()
    except (HTTPError, URLError) as exc:
        print(f"  [warn] fetch failed: {url} -- {exc}")
        return None


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

def fetch_pubmed_abstracts(query: str) -> list[dict]:
    slug = _slug(query)
    cache_file = CORPUS_RAW_DIR / f"pubmed_{slug}.json"

    if cache_file.exists():
        print(f"  [cache] {cache_file.name}")
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    # ESearch -- get PMIDs
    search_params = urlencode({
        "db": "pubmed", "term": query, "retmax": 20, "retmode": "json",
    })
    raw = _fetch_url(f"{PUBMED_BASE}esearch.fcgi?{search_params}")
    time.sleep(0.4)
    if raw is None:
        return []

    pmids = json.loads(raw).get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    # EFetch -- get XML abstracts
    fetch_params = urlencode({
        "db": "pubmed", "id": ",".join(pmids),
        "rettype": "abstract", "retmode": "xml",
    })
    xml_bytes = _fetch_url(f"{PUBMED_BASE}efetch.fcgi?{fetch_params}")
    time.sleep(0.4)
    if xml_bytes is None:
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        print(f"  [warn] XML parse error for query {query!r}: {exc}")
        return []

    docs: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//PubDate/Year")
        author_el = article.find(".//AuthorList/Author/LastName")

        # Structured abstracts have multiple AbstractText elements -- join them
        abstract_els = article.findall(".//AbstractText")
        abstract = " ".join(
            "".join(el.itertext()).strip()
            for el in abstract_els
            if "".join(el.itertext()).strip()
        )
        if not abstract:
            continue

        pmid = pmid_el.text if pmid_el is not None else ""
        docs.append({
            "id": f"pubmed_{pmid}",
            "text": abstract,
            "source": "pubmed",
            "title": "".join(title_el.itertext()).strip() if title_el is not None else "",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "year": year_el.text if year_el is not None else "",
            "topic": query,
            "journal": journal_el.text if journal_el is not None else "",
            "first_author": author_el.text if author_el is not None else "",
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    return docs


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

def fetch_wikipedia_summary(title: str) -> dict | None:
    raw = _fetch_url(f"{WIKI_BASE}page/summary/{title}")
    time.sleep(0.4)
    if raw is None:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    extract = data.get("extract", "").strip()
    if not extract:
        return None

    wiki_url = (
        data.get("content_urls", {}).get("desktop", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{title}"
    )
    return {
        "id": f"wiki_{title}_summary",
        "text": extract,
        "source": "wikipedia",
        "title": data.get("title", title.replace("_", " ")),
        "url": wiki_url,
        "year": "2024",
        "topic": title.replace("_", " "),
    }


def _clean_wikitext(text: str) -> str:
    """Strip wiki markup to get plain text suitable for embedding."""
    # Remove file/image embeds
    text = re.sub(r"\[\[(?:File|Image|Media):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    # [[link|display]] -> display; [[link]] -> link
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", text)
    # Remove templates {{...}} (one level deep — sufficient for most prose sections)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # Remove <ref>...</ref> and self-closing <ref ... />
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove bold/italic wiki markers
    text = text.replace("'''", "").replace("''", "")
    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_wikipedia_sections(title: str) -> list[dict]:
    """Fetch page sections via the MediaWiki Action API (wikitext).
    mobile-sections was decommissioned (T328036); Action API is the stable alternative.
    """
    wiki_url = f"https://en.wikipedia.org/wiki/{title}"
    params = urlencode({
        "action": "parse", "page": title,
        "prop": "wikitext", "format": "json",
    })
    raw = _fetch_url(f"https://en.wikipedia.org/w/api.php?{params}")
    time.sleep(0.4)
    if raw is None:
        return []

    try:
        data = json.loads(raw)
        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
    except (json.JSONDecodeError, KeyError):
        return []

    if not wikitext:
        return []

    # Split on == headings ==
    section_re = re.compile(r"^(==+)\s*(.+?)\s*\1\s*$", re.MULTILINE)
    matches = list(section_re.finditer(wikitext))
    positions = [m.start() for m in matches] + [len(wikitext)]

    docs: list[dict] = []
    for i, match in enumerate(matches):
        section_title = match.group(2)
        body = wikitext[match.end():positions[i + 1]]
        clean = _clean_wikitext(body)
        if len(clean.split()) <= 100:
            continue
        section_slug = _slug(section_title)[:30]
        docs.append({
            "id": f"wiki_{title}_{section_slug}",
            "text": clean,
            "source": "wikipedia",
            "title": f"{title.replace('_', ' ')}: {section_title}",
            "url": wiki_url,
            "year": "2024",
            "topic": title.replace("_", " "),
        })

    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    CORPUS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.\n")

    import chromadb
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{COLLECTION_NAME}' ready. Existing docs: {collection.count()}\n")

    existing_ids: set[str] = set()
    if collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])

    # ---- Collect all documents ----
    all_docs: list[dict] = []

    print("--- Fetching PubMed abstracts ---")
    for query in PUBMED_QUERIES:
        print(f"Query: {query!r}")
        docs = fetch_pubmed_abstracts(query)
        print(f"  -> {len(docs)} abstracts\n")
        all_docs.extend(docs)

    print("--- Fetching Wikipedia articles ---")
    wiki_docs: list[dict] = []

    for title in WIKIPEDIA_ARTICLES:
        doc = fetch_wikipedia_summary(title)
        if doc:
            wiki_docs.append(doc)
            print(f"  -> Summary: {title}")

    print()
    for title in WIKIPEDIA_SECTION_ARTICLES:
        sections = fetch_wikipedia_sections(title)
        wiki_docs.extend(sections)
        print(f"  -> {len(sections)} sections from: {title}")

    wiki_cache = CORPUS_RAW_DIR / "wikipedia.json"
    with open(wiki_cache, "w", encoding="utf-8") as f:
        json.dump(wiki_docs, f, indent=2, ensure_ascii=False)
    print(f"\nWikipedia raw data saved to {wiki_cache}")
    all_docs.extend(wiki_docs)

    # Deduplicate by id (same PMID can appear in multiple query results)
    seen: set[str] = set()
    unique_docs: list[dict] = []
    for doc in all_docs:
        if doc["id"] not in seen:
            seen.add(doc["id"])
            unique_docs.append(doc)

    to_embed = [d for d in unique_docs if d["id"] not in existing_ids]
    print(f"\nTotal unique documents: {len(unique_docs)}")
    print(f"Already in ChromaDB:   {len(unique_docs) - len(to_embed)}")
    print(f"To embed now:          {len(to_embed)}\n")

    pubmed_count = 0
    wiki_count = 0

    for i, doc in enumerate(to_embed, 1):
        embedding = model.encode(doc["text"]).tolist()
        metadata = {k: str(v) for k, v in doc.items() if k not in ("id", "text")}
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["text"]],
            metadatas=[metadata],
        )
        if doc["source"] == "pubmed":
            pubmed_count += 1
        else:
            wiki_count += 1
        print(f"[{i}/{len(to_embed)}] Embedded: {doc['title'][:60]}")

    total = collection.count()
    print(f"\nDone. {total} documents embedded ({pubmed_count} PubMed, {wiki_count} Wikipedia embedded this run).")


if __name__ == "__main__":
    main()
