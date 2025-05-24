import pathlib
import random
import re
import requests
import shutil
import time

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from readability import Document
from tqdm import tqdm

url_files = [
    "urls_5.txt",
    "urls_6.txt",
]

UNPROCESSED_DIR = pathlib.Path("data_web_scrapping/urls/unprocessed")
PROCESSED_DIR = pathlib.Path("data_web_scrapping/urls/processed")
OUT_DIR = pathlib.Path("data_web_scrapping/data")
MIN_WORDS = 3
MAX_WORDS = 40
USER_AGENT = (
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) "
    "Gecko/20100101 Firefox/124.0"
)
# -------------------------------------------------------------------------

HEAD_CLEAN_RE = re.compile(r"[\r\n\t]+")
WHITE_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\b\d+\b")


def load_urls_from_file(fname: str) -> list[str]:
    p = UNPROCESSED_DIR / fname
    with open(p, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def clean_text(txt: str) -> str:
    txt = HEAD_CLEAN_RE.sub(" ", txt)
    txt = WHITE_RE.sub(" ", txt).strip()
    txt = DIGIT_RE.sub("", txt)
    txt = txt.replace('[', '').replace(']', '')
    for quote in ['"', "'", '“', '”', '‘', '’', '«', '»']:
        txt = txt.replace(quote, '')
    return txt.strip()


def split_into_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])", text)
    sentences = []
    for s in raw:
        s_clean = clean_text(s)
        if s_clean.count(':') > 2:
            continue
        if MIN_WORDS <= len(s_clean.split()) <= MAX_WORDS:
            sentences.append(s_clean)
    return sentences


def fetch_article(url: str) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
    except Exception as exc:
        print(f"[WARN] {url} – {exc}")
        return None

    doc = Document(r.text)
    html = doc.summary()
    soup = BeautifulSoup(html, "html.parser")
    for sup in soup.find_all('sup'):
        sup.decompose()
    text = soup.get_text(" ")

    if not text.strip():
        soup_full = BeautifulSoup(r.text, "html.parser")
        article = soup_full.find('article') or soup_full
        for sup in article.find_all('sup'):
            sup.decompose()
        text = article.get_text(" ")

    return clean_text(text)


for url_file in url_files:
    all_rows: list[str] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    urls_to_scrape = load_urls_from_file(url_file)
    if not urls_to_scrape:
        print(f"[SKIP] No urls in file: {url_file}.")
        continue

    for url in tqdm(urls_to_scrape, desc="Scraping"):
        netloc = urlparse(url).netloc.replace('www.', '')
        target_dir = OUT_DIR / "raw" / netloc
        target_dir.mkdir(parents=True, exist_ok=True)

        art_text = fetch_article(url)
        if not art_text:
            continue

        sentences = split_into_sentences(art_text)
        if not sentences:
            print(f"[WARN] No valuable sentences for: {url}")
            continue

        timestamp = int(time.time())
        fname = target_dir / f"{netloc}_{timestamp}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

        print(f"[INFO] Source has been scrapped ({len(sentences)} sentences): {url}.")

        all_rows.extend(sentences)
        time.sleep(random.uniform(1, 3))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(UNPROCESSED_DIR / url_file), str(PROCESSED_DIR / url_file))
    print(f"[INFO] File {url_file} has been processed.")

    if all_rows:
        summary_dir = OUT_DIR / "unrefined"
        summary_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        summary_file = summary_dir / f"processed_{timestamp}.txt"

        print(f"[INFO] Total numer of sentences: {len(all_rows)}.")

        all_rows = list(set(all_rows))

        print(f"[INFO] Total numer of sentences (without duplication): {len(all_rows)}.")

        with open(summary_file, "w", encoding="utf-8") as f:
            for s in all_rows:
                f.write(s + "\n")
        print(f"[FINISH] Total {len(all_rows)} sentences to {summary_file} have been saved.")

    else:
        print("[WARN] No sentences have been collected.")
