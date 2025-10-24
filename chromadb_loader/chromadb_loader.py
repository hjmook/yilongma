import chromadb
import json
from typing import List, Dict, Optional
import re
import os

class ElonKnowledgeBaseLoader:
    """
    Loads Elon Musk knowledge base into ChromaDB with automatic chunking.
    
    ChromaDB is a vector database - it stores text chunks as embeddings (numerical representations)
    that allow semantic search. When you query it, it finds chunks that are conceptually similar
    to your query, not just keyword matches.
    """
    
    def __init__(self, persist_directory="./chroma_db", reset=False):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Where to save the database on disk
            reset: If True, deletes existing collection and starts fresh
        """
        # Create a persistent ChromaDB client (data saved to disk)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Reset if requested
        if reset:
            try:
                self.client.delete_collection(name="elon_musk_knowledge")
                print("-> Deleted existing collection")
            except:
                pass
        
        # Create or get collection (like a table in traditional DBs)
        # Collections store related documents together
        self.collection = self.client.get_or_create_collection(
            name="elon_musk_knowledge",
            metadata={"description": "Elon Musk knowledge base for RAG"}
        )
        
        print(f"\n{'=' * 70}")
        print(f"ChromaDB initialized at {persist_directory}")
        print(f"  Collection: {self.collection.name}")
        print(f"  Current document count: {self.collection.count()}")
        print(f"{'=' * 70}\n")
    
    def split_into_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Args:
            text: Full document text
            chunk_size: Approximate tokens per chunk (1 token ~= 4 chars)
            overlap: Characters to overlap between chunks
        
        Returns:
            List of text chunks
        """
        # Rough conversion: 1 token ~= 4 characters
        char_limit = chunk_size * 4
        
        # Split by sentences first (better than arbitrary cuts)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds limit, start new chunk
            if len(current_chunk) + len(sentence) > char_limit and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap (last few chars of previous)
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk += " " + sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_existing_doc_ids(self) -> set:
        """Get set of existing document IDs to avoid duplicates"""
        try:
            if self.collection.count() == 0:
                return set()
            
            # Get all existing IDs
            existing = self.collection.get()
            existing_ids = existing['ids']
            
            # Extract parent doc IDs (format: doc_X_chunk_Y)
            parent_ids = set()
            for id_str in existing_ids:
                if '_chunk_' in id_str:
                    parent_id = id_str.split('_chunk_')[0]
                    parent_ids.add(parent_id)
            
            return parent_ids
        except Exception as e:
            print(f"Warning: Could not get existing IDs: {e}")
            return set()
    
    def load_documents(self, json_file: str = 'elon_musk_knowledge_base.json', skip_duplicates=True):
        """
        Load documents from JSON and add to ChromaDB with chunking.
        
        Args:
            json_file: Path to the scraped data JSON file
            skip_duplicates: If True, skip documents already in DB
        """
        if not os.path.exists(json_file):
            print(f"ERROR: File not found: {json_file}")
            return
        
        print(f"Loading documents from {json_file}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"Found {len(documents)} documents in JSON file\n")
        
        # Get existing doc IDs if skip_duplicates is enabled
        existing_ids = set()
        if skip_duplicates:
            existing_ids = self.get_existing_doc_ids()
            print(f"Already have {len(existing_ids)} documents in DB")
            print(f"Will skip duplicates\n")
        
        total_chunks = 0
        skipped_docs = 0
        failed_docs = []
        
        for idx, doc in enumerate(documents):
            try:
                # Create unique document ID
                doc_id = f"doc_{idx}"
                
                # Skip if already exists
                if skip_duplicates and doc_id in existing_ids:
                    skipped_docs += 1
                    continue
                
                # Validate required fields
                if not all(k in doc for k in ['content', 'date', 'source']):
                    failed_docs.append({
                        'index': idx,
                        'reason': 'Missing required fields',
                        'doc': doc.get('metadata', {}).get('title', 'Unknown')
                    })
                    continue
                
                # Validate content length
                if len(doc['content']) < 100:
                    failed_docs.append({
                        'index': idx,
                        'reason': 'Content too short',
                        'doc': doc.get('metadata', {}).get('title', 'Unknown')
                    })
                    continue
                
                # Split into chunks
                chunks = self.split_into_chunks(doc['content'])
                
                # Add each chunk to ChromaDB
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                    
                    # Metadata for this chunk (ChromaDB uses this for filtering)
                    metadata = {
                        'date': doc['date'],
                        'source': doc['source'],
                        'parent_doc': doc_id,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'title': doc.get('metadata', {}).get('title', ''),
                        'topic': doc.get('metadata', {}).get('topic', ''),
                        'url': doc.get('metadata', {}).get('url', '')
                    }
                    
                    # Add to ChromaDB
                    # ChromaDB automatically creates embeddings from the text
                    self.collection.add(
                        ids=[chunk_id],
                        documents=[chunk],
                        metadatas=[metadata]
                    )
                
                total_chunks += len(chunks)
                
                if (idx + 1) % 50 == 0:
                    print(f"  Progress: {idx + 1}/{len(documents)} documents processed...")
                
            except Exception as e:
                failed_docs.append({
                    'index': idx,
                    'reason': str(e),
                    'doc': doc.get('metadata', {}).get('title', 'Unknown')
                })
        
        print(f"\n{'=' * 70}")
        print(f"LOADING COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Documents in JSON: {len(documents)}")
        print(f"  Documents skipped (duplicates): {skipped_docs}")
        print(f"  New chunks created: {total_chunks}")
        print(f"  Failed documents: {len(failed_docs)}")
        print(f"  Total ChromaDB collection size: {self.collection.count()}")
        print(f"{'=' * 70}")
        
        if failed_docs:
            print(f"\nWarning: {len(failed_docs)} documents failed to load:")
            for fail in failed_docs[:5]:  # Show first 5
                print(f"  - Doc {fail['index']}: {fail['reason']}")
            if len(failed_docs) > 5:
                print(f"  ... and {len(failed_docs) - 5} more")
    
    def query(self, query_text: str, n_results: int = 5, filter_metadata: Optional[Dict] = None):
        """
        Query the knowledge base (semantic search).
        
        Args:
            query_text: Your question/query
            n_results: How many relevant chunks to return
            filter_metadata: Optional filters, e.g., {'source': 'tesla.com'}
        
        Returns:
            Query results with documents and metadata
        """
        if self.collection.count() == 0:
            print("ERROR: Collection is empty. Load documents first.")
            return None
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata  # Optional: filter by metadata
        )
        return results
    
    def get_stats(self):
        """Get statistics about the knowledge base"""
        total = self.collection.count()
        
        if total == 0:
            print("\nCollection is empty. Load documents first.")
            return
        
        # Get sample to analyze
        sample = self.collection.get(limit=min(total, 10000))  # Cap at 10k for performance
        
        # Count by source, date, topic
        sources = {}
        dates = {}
        topics = {}
        years = {}
        
        for metadata in sample['metadatas']:
            source = metadata.get('source', 'unknown')
            date = metadata.get('date', 'unknown')
            topic = metadata.get('topic', 'unknown')
            
            sources[source] = sources.get(source, 0) + 1
            dates[date] = dates.get(date, 0) + 1
            topics[topic] = topics.get(topic, 0) + 1
            
            # Extract year
            if date != 'unknown' and len(date) >= 4:
                year = date[:4]
                years[year] = years.get(year, 0) + 1
        
        print(f"\n{'=' * 70}")
        print("KNOWLEDGE BASE STATISTICS")
        print(f"{'=' * 70}")
        print(f"Total chunks: {total:,}")
        
        print(f"\nTop 10 Sources:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {source}: {count:,} chunks")
        
        print(f"\nTop 10 Topics:")
        for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {topic}: {count:,} chunks")
        
        print(f"\nYear Distribution:")
        for year, count in sorted(years.items(), reverse=True):
            print(f"  {year}: {count:,} chunks")
        
        print(f"{'=' * 70}")
    
    def test_query(self, query: str):
        """Test a query and display results nicely"""
        print(f"\n{'=' * 70}")
        print("QUERY TEST")
        print(f"{'=' * 70}")
        print(f"Query: '{query}'")
        print(f"\nTop 3 relevant chunks:\n")
        
        results = self.query(query_text=query, n_results=3)
        
        if not results:
            return
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"{i+1}. [{metadata['date']}] {metadata['source']}")
            if metadata.get('title'):
                print(f"   Title: {metadata['title']}")
            print(f"   Preview: {doc[:250]}...")
            print()
        
        print(f"{'=' * 70}")


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ELON MUSK KNOWLEDGE BASE - CHROMADB LOADER")
    print("=" * 70)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(script_dir)
    
    # Build paths relative to project root
    kb_path = os.path.join(project_root, 'elon_musk_knowledge_base.json')
    db_path = os.path.join(project_root, 'elon_chroma_db')
    
    # Initialize loader
    # Set reset=True to start fresh, False to add to existing
    loader = ElonKnowledgeBaseLoader(
        persist_directory=db_path,
        reset=False  # Change to True to start fresh
    )
    
    # Load your scraped data
    loader.load_documents(kb_path, skip_duplicates=True)
    
    # View statistics
    loader.get_stats()
    
    # Example queries
    loader.test_query("What is Elon Musk doing with the Department of Government Efficiency?")
    loader.test_query("What are the latest SpaceX Starship developments?")
    loader.test_query("Tell me about Tesla's Cybertruck production")