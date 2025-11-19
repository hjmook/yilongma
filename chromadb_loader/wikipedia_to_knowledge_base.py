import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import quote_plus

WIKI_DIR = os.path.join(os.path.dirname(__file__), 'wikipedia_pages')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'elon_musk_knowledge_base.json')
API_URL = 'https://en.wikipedia.org/w/api.php'

# Minimum chunk length
MIN_CHARS = 100
CHUNK_SIZE = 2048  # Target chunk size (matches 512 tokens)
CHUNK_OVERLAP = 50


def get_last_revision_date(title):
    params = {
        'action': 'query',
        'prop': 'revisions',
        'rvprop': 'timestamp',
        'titles': title,
        'format': 'json',
        'redirects': True
    }
    headers = {
        'User-Agent': 'ElonMuskKnowledgeBaseBot/1.0 (contact: your_email@example.com)'
    }
    try:
        r = requests.get(API_URL, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        pages = data.get('query', {}).get('pages', {})
        for pageid, page in pages.items():
            revs = page.get('revisions')
            if revs:
                ts = revs[0]['timestamp']
                # Format: 2025-10-25T12:34:56Z -> 2025-10-25
                return ts[:10]
    except Exception:
        pass
    return '2025-01-01'  # Fallback snapshot date



import re
def chunk_sentences(text):
    # Split text into sentences, then group into ~CHUNK_SIZE char chunks with CHUNK_OVERLAP
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > CHUNK_SIZE and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: last CHUNK_OVERLAP chars
            current_chunk = current_chunk[-CHUNK_OVERLAP:] + " " + sentence
        else:
            current_chunk += " " + sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c) >= MIN_CHARS]


def clean_text(text):
    # Remove citation markers [1], [2], etc.
    import re
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text)
    text = re.sub(r'\[edit\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def process_wikipedia_page(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    title = data['title']
    html = data['html']
    url_title = title.replace(' ', '_')
    url = f'https://en.wikipedia.org/wiki/{quote_plus(url_title)}'
    topic = 'Biography/Reference' if title == 'Elon Musk' else title
    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')
    parser_output = soup.find('div', class_='mw-parser-output')
    if not parser_output:
        return []
    # Get all paragraphs (ignore tables, infoboxes, etc.)
    paragraphs = []
    for elem in parser_output.find_all(['p'], recursive=False):
        text = elem.get_text().strip()
        if len(text) >= 40:
            paragraphs.append(clean_text(text))
    if not paragraphs:
        return []
    # Join all paragraphs for a single full-document content
    full_text = ' '.join(paragraphs)
    # Get date
    date = get_last_revision_date(title)
    # Build single doc
    doc = {
        'content': full_text,
        'date': date,
        'source': 'wikipedia.org',
        'metadata': {
            'title': title,
            'topic': topic,
            'url': url
        }
    }
    return [doc]


def main():
    print(f'Processing Wikipedia pages from: {WIKI_DIR}')
    # Load existing KB
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            kb_docs = json.load(f)
    else:
        kb_docs = []
    # Build dedup sets from existing KB
    seen_hashes = set(hashlib.sha256(doc['content'][:500].encode('utf-8')).hexdigest() for doc in kb_docs)
    seen_urls = set(doc.get('metadata', {}).get('url', '') for doc in kb_docs)
    new_docs = []
    for fname in os.listdir(WIKI_DIR):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(WIKI_DIR, fname)
        docs = process_wikipedia_page(path)
        added = 0
        for doc in docs:
            url = doc['metadata']['url']
            content_hash = hashlib.sha256(doc['content'][:500].encode('utf-8')).hexdigest()
            if url in seen_urls or content_hash in seen_hashes:
                continue
            seen_urls.add(url)
            seen_hashes.add(content_hash)
            kb_docs.append(doc)
            new_docs.append(doc)
            added += 1
        print(f'  {fname}: {added} new docs added')
    print(f'\nTotal new Wikipedia docs added: {len(new_docs)}')
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(kb_docs, f, ensure_ascii=False, indent=2)
    print(f'Saved to {OUTPUT_FILE}')

if __name__ == '__main__':
    main()
