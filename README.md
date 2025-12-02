# meta-enriched-rag-for-legal-llms
Repository for 685: Advanced NLP course project — Meta-Enriched RAG for Legal LLMs. Contains all code files, setup scripts, and experimental reports for developing and evaluating a metadata-enriched, hybrid retrieval-augmented generation (RAG) framework tailored for legal language models.

---

## Chunking

This project supports two chunking strategies for preparing `.txt` documents for RAG:

### **1. Fixed Chunking (Sliding Window)**

**Function:** `chunk_folder_llama_json`

* Splits text purely by **token count** using a sliding window.
* Parameters:

  * `chunk_size` = max tokens per chunk
  * `window` = token overlap
* Uses tokenizer offset mappings to map token spans back to original text.
* Produces evenly sized, deterministic chunks, but **does not respect text structure**.

**Usage:**

```python
chunk_folder_llama_json(
    input_folder="...",
    output_json="chunks_fixed.json",
    chunk_size=500,
    window=50
)
```

---

### **2. Recursive Chunking (Structure-Aware)**

**Function:** `chunk_folder_llama_json_recursive`

* Tries to split text at **natural boundaries** (paragraph → sentence → word).
* If the segment is still too large, falls back to **token-based splitting**.
* After segmentation, merges pieces into final chunks with token overlap.
* Produces more coherent chunks for semantic retrieval.

**Usage:**

```python
chunk_folder_llama_json_recursive(
    input_folder="...",
    output_json="chunks_recursive.json",
    chunk_size=500,
    window=50
)
```

---

### **Output Format**

Both methods produce a single JSON:

```json
{
  "folder_filename_chunk1": {
    "chunk_text": "...",
    "span": [char_start, char_end]
  }
}
```

---

### **Important**

After switching chunking modes or parameters, always **rebuild embeddings + rerun retrieval**, otherwise evaluation may reference chunk IDs that no longer exist.

---

