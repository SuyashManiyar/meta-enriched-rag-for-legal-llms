"""Document loaders for various formats"""

from pathlib import Path
from typing import List, Dict


class DocumentLoader:
    """Load documents from various formats"""
    
    @staticmethod
    def load_txt(filepath: Path) -> str:
        """Load text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_all(directory: str, extensions: List[str] = ['.txt']) -> List[Dict]:
        """Load all documents from directory"""
        documents = []
        path = Path(directory)
        
        for ext in extensions:
            for file_path in path.glob(f"*{ext}"):
                text = DocumentLoader.load_txt(file_path)
                documents.append({
                    'text': text,
                    'metadata': {
                        'source': file_path.name,
                        'id': file_path.stem
                    }
                })
        
        return documents
