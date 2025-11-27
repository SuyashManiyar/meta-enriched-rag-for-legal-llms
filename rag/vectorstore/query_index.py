"""Query vector index"""

import sqlite3
from typing import List, Dict
import sqlite_vec
from .embedder import Embedder


class IndexQuerier:
    """Query vector index for retrieval"""
    
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
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k documents"""
        self._connect()
        
        query_embedding = self.embedder.embed_query(query)
        
        cursor = self.conn.execute(f"""
            SELECT 
                d.id,
                d.text,
                d.metadata,
                distance
            FROM vec_documents v
            JOIN documents d ON v.id = d.id
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """, (query_embedding.tobytes(), top_k))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'text': row[1],
                'metadata': row[2],
                'distance': row[3]
            })
        
        return results
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
