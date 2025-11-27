"""Text cleaning and normalization"""

import re
from typing import List, Dict


class TextCleaner:
    """Clean and normalize text"""
    
    @staticmethod
    def clean(text: str) -> str:
        """Clean text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep legal punctuation
        text = re.sub(r'[^\w\s.,;:!?()\[\]§-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def clean_documents(documents: List[Dict]) -> List[Dict]:
        """Clean all documents"""
        cleaned = []
        for doc in documents:
            cleaned.append({
                'text': TextCleaner.clean(doc['text']),
                'metadata': doc['metadata']
            })
        return cleaned
