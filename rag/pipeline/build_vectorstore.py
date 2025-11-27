"""Build vector store: embed → index"""

import sys
sys.path.append('..')

from vectorstore.build_index import IndexBuilder
from utils.file_utils import load_json, ensure_dir
from utils.timing import timing_decorator
import yaml


@timing_decorator
def build_vectorstore(config_path: str = "../config/rag_config.yaml"):
    """Build vector store pipeline"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open("../config/paths.yaml", 'r') as f:
        paths = yaml.safe_load(f)
    
    # Load chunks
    print("=== Loading Chunks ===")
    data = load_json(f"{paths['data']['chunks']}/chunks.json")
    chunks = data['chunks']
    print(f"Loaded {len(chunks)} chunks")
    
    # Build index
    print("\n=== Building Vector Index ===")
    ensure_dir(paths['data']['index'])
    db_path = f"{paths['data']['index']}/vectors.db"
    
    builder = IndexBuilder(
        db_path=db_path,
        embedding_model=config['embedding']['model_name']
    )
    
    builder.build(chunks)
    builder.close()
    
    print(f"\n✓ Vector store saved to {db_path}")


if __name__ == "__main__":
    build_vectorstore()
