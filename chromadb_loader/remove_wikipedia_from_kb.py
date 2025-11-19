import json
import os

KB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'elon_musk_knowledge_base.json')
BACKUP_PATH = KB_PATH + '.bak'

with open(KB_PATH, 'r', encoding='utf-8') as f:
    docs = json.load(f)

# Backup original
with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

filtered = [doc for doc in docs if doc.get('source') != 'wikipedia.org']
removed = len(docs) - len(filtered)

with open(KB_PATH, 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"Removed {removed} Wikipedia docs. Backup saved as {BACKUP_PATH}.")
