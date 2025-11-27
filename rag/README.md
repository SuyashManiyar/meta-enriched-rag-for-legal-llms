# Production RAG Pipeline for Legal LLMs

Complete RAG pipeline for evaluating LLaMA 3.2 on MUAD legal corpus.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Corpus
Load, clean, and chunk documents:
```bash
python main.py build
```

### 3. Build Vector Store
Generate embeddings and create index:
```bash
python main.py index
```

### 4. Query
```bash
python main.py query --query "What are the key terms in merger agreements?" --top-k 5
```

## Configuration

Edit YAML files in `config/`:
- `rag_config.yaml` - Chunking, embedding, retrieval settings
- `model_config.yaml` - LLaMA model and generation parameters
- `paths.yaml` - All data paths

## Structure

```
rag/
├── config/              # YAML configurations
├── preprocessing/       # Load, clean, chunk
├── vectorstore/         # Embed and index
├── generator/           # LLaMA inference
├── pipeline/            # End-to-end workflows
├── utils/               # Helpers
└── main.py             # CLI entry point
```

## Usage Examples

```python
from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.query("Your question here")
print(result['answer'])
pipeline.close()
```
