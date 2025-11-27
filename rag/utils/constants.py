"""Constants for RAG pipeline"""

# File extensions
SUPPORTED_TEXT_FORMATS = ['.txt', '.json']
SUPPORTED_DOC_FORMATS = ['.pdf', '.docx', '.doc']

# Chunking strategies
CHUNK_STRATEGY_FIXED = "fixed"
CHUNK_STRATEGY_RECURSIVE = "recursive"
CHUNK_STRATEGY_SEMANTIC = "semantic"

# Index types
INDEX_FAISS = "faiss"
INDEX_HNSW = "hnsw"
INDEX_SQLITE_VEC = "sqlite-vec"

# Reranker types
RERANKER_COHERE = "cohere"
RERANKER_JINA = "jina"
RERANKER_LOCAL = "local"
