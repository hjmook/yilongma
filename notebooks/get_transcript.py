import requests
from bs4 import BeautifulSoup
import re
import json
import os

def download_and_parse_transcript(url):
    print("Downloading transcript...")
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove script/style elements
    for script in soup(["script", "style", "nav", "header", "footer"]):
        script.decompose()

    # Focus on main content
    main = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'content|post|entry', re.I))
    if main is None:
        main = soup.body

    return main.get_text(separator='\n', strip=True)

def clean_and_structure_transcript(raw_text):
    lines = raw_text.split('\n')
    cleaned = []

    # Patterns
    timestamp_pattern = re.compile(r'^\d+:\d+\s*–')  # For ToC lines like "0:07 – War..."
    speaker_pattern = re.compile(r'^(Lex Fridman|Elon Musk)\s*$', re.IGNORECASE)
    inaudible_pattern = re.compile(r'\[inaudible.*?\]')
    url_pattern = re.compile(r'https?://\S+')
    email_pattern = re.compile(r'\S+@\S+')
    extra_whitespace = re.compile(r'\s+')
    inline_timestamp_pattern = re.compile(r'^\(\d{2}:\d{2}:\d{2}\)\s*')  # e.g., (00:06:19)

    current_speaker = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip metadata lines
        if any(skip in line.lower() for skip in [
            'transcript of', 'clickable links', 'human generated', 'watch the full',
            'go back to', 'here are some useful', 'table of contents', 'jump approximately',
            'introduction', '##'
        ]):
            continue

        # Remove inline timestamps like (00:06:19)
        line = inline_timestamp_pattern.sub('', line)

        # Skip URLs, emails
        if url_pattern.search(line) or email_pattern.search(line):
            continue

        # Clean inaudible tags
        line = inaudible_pattern.sub('[inaudible]', line)
        line = extra_whitespace.sub(' ', line).strip()

        if not line or line == '[inaudible]':
            continue

        # Detect speaker change
        speaker_match = speaker_pattern.match(line)
        if speaker_match:
            current_speaker = speaker_match.group(1).strip()
            continue

        # Record utterance if speaker is known
        if current_speaker and line:
            cleaned.append({
                "speaker": current_speaker,
                "text": line
                # No "section" field — sections removed as requested
            })

    return cleaned

def save_to_jsonl(data, output_path="elon_musk_4_transcript_clean.jsonl"):
    print(f"Saving {len(data)} utterances to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print("Done.")

def main():
    # url = "https://lexfridman.com/elon-musk-4-transcript"
    # url = "https://singjupost.com/full-transcript-elon-musk-on-joe-rogan-podcast/"
    url = "https://www.cnbc.com/2025/05/20/cnbc-transcript-elon-musk-sits-down-with-cnbcs-david-faber-live-on-cnbc-today-.html"
    raw = download_and_parse_transcript(url)
    structured = clean_and_structure_transcript(raw)
    save_to_jsonl(structured)

if __name__ == "__main__":
    main()