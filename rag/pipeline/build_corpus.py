"""Build corpus: load → clean → chunk"""

import sys
sys.path.append('..')

from preprocessing.loader import DocumentLoader
from preprocessing.cleaner import TextCleaner
from preprocessing.chunker import TextChunker
from utils.file_utils import save_json, ensure_dir
from utils.timing import timing_decorator
import yaml


@timing_decorator
def build_corpus(config_path: str = "../config/rag_config.yaml"):
    """Build corpus pipeline"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open("../config/paths.yaml", 'r') as f:
        paths = yaml.safe_load(f)
    
    # Step 1: Load documents
    print("=== Loading Documents ===")
    documents = DocumentLoader.load_all(paths['corpus']['maud'])
    print(f"Loaded {len(documents)} documents")
    
    # Step 2: Clean documents
    print("\n=== Cleaning Documents ===")
    cleaned_docs = TextCleaner.clean_documents(documents)
    ensure_dir(paths['data']['cleaned'])
    save_json({'documents': cleaned_docs}, f"{paths['data']['cleaned']}/cleaned_docs.json")
    print(f"Cleaned {len(cleaned_docs)} documents")
    
    # Step 3: Chunk documents
    print("\n=== Chunking Documents ===")
    chunker = TextChunker(
        chunk_size=config['chunking']['chunk_size'],
        overlap=config['chunking']['chunk_overlap']
    )
    chunked_docs = chunker.chunk_documents(cleaned_docs)
    ensure_dir(paths['data']['chunks'])
    save_json({'chunks': chunked_docs}, f"{paths['data']['chunks']}/chunks.json")
    print(f"Created {len(chunked_docs)} chunks")
    
    return chunked_docs


if __name__ == "__main__":
    build_corpus()
