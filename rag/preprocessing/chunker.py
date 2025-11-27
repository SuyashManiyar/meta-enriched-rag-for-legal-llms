"""Text chunking strategies"""

from typing import List, Dict


class TextChunker:
    """Chunk text into smaller pieces"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk single text"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk all documents"""
        chunked = []
        
        for doc in documents:
            text = doc['text']
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                chunked.append({
                    'text': chunk,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                })
        
        return chunked
