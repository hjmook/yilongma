import requests
import os
import json
from time import sleep

# List of Wikipedia page titles to download
PAGES = [
    'Elon Musk',
    'Musk family',
    'List of children of Elon Musk',
    'Tesla, Inc.',
    'SpaceX',
    'Neuralink',
    'The Boring Company',
    'X.AI',
    'Twitter, Inc.',
    'PayPal',
    'Zip2',
    'SolarCity',
    'OpenAI',
    'Falcon 9',
    'Starship (spacecraft)',
    'Tesla Model S',
    'Tesla Model 3',
    'Tesla Cybertruck',
    'Starlink',
    'Walter Isaacson',
    'PayPal Mafia',
    'Grimes',
    'Justine Musk',
    'Talulah Riley',
]

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'wikipedia_pages')
os.makedirs(SAVE_DIR, exist_ok=True)

API_URL = 'https://en.wikipedia.org/w/api.php'


def fetch_html(title):
    params = {
        'action': 'parse',
        'page': title,
        'format': 'json',
        'prop': 'text',
        'redirects': True
    }
    headers = {
        'User-Agent': 'ElonMuskKnowledgeBaseBot/1.0 (contact: your_email@example.com)'
    }
    r = requests.get(API_URL, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    if 'parse' in data and 'text' in data['parse']:
        return data['parse']['text']['*']
    return None

def fetch_wikitext(title):
    params = {
        'action': 'query',
        'prop': 'revisions',
        'rvprop': 'content',
        'rvslots': 'main',
        'titles': title,
        'format': 'json',
        'redirects': True
    }
    headers = {
        'User-Agent': 'ElonMuskKnowledgeBaseBot/1.0 (contact: your_email@example.com)'
    }
    r = requests.get(API_URL, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    pages = data.get('query', {}).get('pages', {})
    for pageid, page in pages.items():
        revs = page.get('revisions')
        if revs:
            return revs[0]['slots']['main']['*']
    return None

def save_page(title, html, wikitext):
    safe_title = title.replace('/', '_')
    out = {
        'title': title,
        'html': html,
        'wikitext': wikitext
    }
    with open(os.path.join(SAVE_DIR, f'{safe_title}.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main():
    print(f"Saving Wikipedia pages to: {SAVE_DIR}\n")
    for title in PAGES:
        print(f"Fetching: {title} ...", end=' ', flush=True)
        try:
            html = fetch_html(title)
            wikitext = fetch_wikitext(title)
            if html or wikitext:
                save_page(title, html, wikitext)
                print("✓ saved")
            else:
                print("✗ not found")
        except Exception as e:
            print(f"✗ error: {e}")
        sleep(1)
    print("\nDone.")

if __name__ == '__main__':
    main()
