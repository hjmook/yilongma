#!/usr/bin/env python3
"""
Simple script to query the ChromaDB vector database and see what chunks are returned.
"""
import chromadb
import sys
import os

# Path copied EXACTLY from fast_model_server.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "elon_chroma_db")

def query_db(query_text, n_results=5):
    """Query the ChromaDB database and display results - EXACT copy of fast_model_server pattern"""
    
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"ERROR: ChromaDB path not found: {CHROMA_DB_PATH}")
        return
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name="elon_musk_knowledge")
    except Exception as e:
        print(f"ERROR: Could not load collection: {e}")
        return
    
    print(f"\n{'=' * 80}")
    print(f"QUERY: {query_text}")
    print(f"{'=' * 80}")
    print(f"Database: {CHROMA_DB_PATH}")
    print(f"Collection: {collection.name}")
    print(f"Requesting top {n_results} results\n")
    
    # Query EXACTLY like fast_model_server.py does it
    try:
        results = collection.query(query_texts=[query_text], n_results=n_results)
    except Exception as e:
        print(f"ERROR during query: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Format results EXACTLY like fast_model_server.py
    if not results or not results.get('documents'):
        print("No results found!")
        return
    
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results.get('distances', [[]])[0]
    
    print(f"Found {len(docs)} results\n")
    
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        print(f"\n{'─' * 80}")
        print(f"RESULT #{i+1}")
        print(f"{'─' * 80}")
        
        # Metadata
        print(f"📅 Date: {meta.get('date', 'Unknown')}")
        print(f"🌐 Source: {meta.get('source', 'Unknown')}")
        print(f"📝 Title: {meta.get('title', '')}")
        print(f"🏷️  Topic: {meta.get('topic', '')}")
        print(f"🔗 URL: {meta.get('url', '')}")
        print(f"📏 Distance: {distances[i] if i < len(distances) else 'N/A'}")
        
        # Chunk info
        if 'chunk_index' in meta:
            print(f"📦 Chunk: {meta['chunk_index'] + 1}/{meta.get('total_chunks', '?')}")
        
        # Content
        print(f"\n📄 CONTENT ({len(doc)} chars):")
        print(f"─" * 80)
        # content_preview = doc[:500] + "..." if len(doc) > 500 else doc
        # print(content_preview)
        print(doc)
    
    print(f"\n{'=' * 80}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python query_chromadb.py 'your query here' [num_results]")
        print("\nExample queries:")
        print("  python query_chromadb.py 'who are Elon Musk's children'")
        print("  python query_chromadb.py 'Tesla Cybertruck production' 10")
        print("  python query_chromadb.py 'SpaceX Starship' 3")
        sys.exit(1)
    
    query_text = sys.argv[1]
    n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    query_db(query_text, n_results)


if __name__ == "__main__":
    main()