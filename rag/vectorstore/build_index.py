"""Build vector index"""

import sqlite3
import numpy as np
from typing import List, Dict
import sqlite_vec
from .embedder import Embedder


class IndexBuilder:
    """Build and manage vector index"""
    
    def __init__(self, db_path: str, embedding_model: str):
        self.db_path = db_path
        self.embedder = Embedder(embedding_model)
        self.conn = None
    
    def _connect(self):
        """Connect to database"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
    
    def build(self, documents: List[Dict]):
        """Build index from documents"""
        self._connect()
        
        # Create tables
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB
            )
        """)
        
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
                id INTEGER PRIMARY KEY,
                embedding FLOAT[{self.embedder.embedding_dim}]
            )
        """)
        
        print(f"Embedding {len(documents)} documents...")
        
        # Process in batches
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            texts = [doc['text'] for doc in batch]
            
            embeddings = self.embedder.embed_texts(texts, batch_size)
            
            for doc, embedding in zip(batch, embeddings):
                cursor = self.conn.execute(
                    "INSERT INTO documents (text, metadata) VALUES (?, ?)",
                    (doc['text'], str(doc.get('metadata', {})))
                )
                doc_id = cursor.lastrowid
                
                self.conn.execute(
                    "INSERT INTO vec_documents (id, embedding) VALUES (?, ?)",
                    (doc_id, embedding.tobytes())
                )
            
            self.conn.commit()
            print(f"Processed {min(i+batch_size, len(documents))}/{len(documents)}")
        
        print("Index built successfully!")
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
