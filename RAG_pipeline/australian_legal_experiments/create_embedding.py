import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from tqdm import tqdm

# =========================
# CONFIG
# =========================
EMBED_MODEL = "thenlper/gte-large"

# Create output directory
OUTPUT_DIR = "australian_legal_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Australian Legal Meta Recursive Chunking - ACTIVE
CHUNK_JSON_PATH = "../../australian_legal_data/generated_chunks/australian_legal_text_meta_recursive_chunking.json"
FAISS_INDEX_PATH = "../embeddings_with_span/australian_legal_faiss_emd_meta_recursive.bin"
META_PATH = "../embeddings_with_span/australlian_legal_faiss_emb_meta_recursive.json"
STATS_PATH = "../embeddings_with_span/australian_legal_token_stats_meta_recursive.json"

# Australian Legal Metadata Enhanced 380 - COMMENTED OUT
# CHUNK_JSON_PATH = "existing_metadata_enhanced_380.json"
# FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "australian_legal_metadata_enhanced_380_faiss.bin")
# META_PATH = os.path.join(OUTPUT_DIR, "australian_legal_metadata_enhanced_380_meta.json")
# STATS_PATH = os.path.join(OUTPUT_DIR, "australian_legal_metadata_enhanced_380_token_stats.json")

# Australian Legal Test Recursive Chunking - COMMENTED OUT
# CHUNK_JSON_PATH = "australian_legal_text_recursive_chunking.json"
# FAISS_INDEX_PATH = "australian_legal_faiss_emb_recursive.bin" 
# META_PATH = "australian_legal_faiss_emb_recursive.json" 
# STATS_PATH = "australian_legal_token_stats_recursive.json"

# OLD MAUD PATHS - COMMENTED OUT
# CHUNK_JSON_PATH = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_chunks/chunks_window_summary_n_doc_name.json" 
# FAISS_INDEX_PATH = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_embs/faiss_emb_recur_w_window_summary_n_doc_name.bin" 
# META_PATH = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_embs/faiss_emb_recur_w_window_summary_n_doc_name.json" 
# STATS_PATH = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_embs/token_stats_recur_w_window_summary_n_doc_name.json"

# OLD PRIVACY QA PATHS - COMMENTED OUT
# CHUNK_JSON_PATH = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_chunks/chunks_window_summary_n_doc_name.json" 
# FAISS_INDEX_PATH = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_embs/faiss_emb_recur_w_window_summary_n_doc_name.bin" 
# META_PATH = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_embs/faiss_emb_recur_w_window_summary_n_doc_name.json" 
# STATS_PATH = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_embs/token_stats_recur_w_window_summary_n_doc_name.json"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16
NORMALIZE = True

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, use_fast=True)
model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
model.eval()


# =========================
# MEAN POOLING
# =========================
def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


# =========================
# ENCODER
# =========================
def encode_texts(text_list):
    out = []
    for i in range(0, len(text_list), BATCH):
        b = text_list[i:i+BATCH]
        # Strict truncation to 512 to prevent RuntimeError
        inputs = tokenizer(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model(**inputs, return_dict=True)
            emb = mean_pool(output.last_hidden_state, inputs["attention_mask"])
        emb = emb.cpu().numpy().astype("float32")
        out.append(emb)

    embs = np.vstack(out)
    if NORMALIZE:
        faiss.normalize_L2(embs)
    return embs


# =========================
# BUILD INDEX (UPDATED)
# =========================
def build_faiss_index():
    # Load chunk list (metadata enhanced format is a list of dicts)
    with open(CHUNK_JSON_PATH, "r", encoding="utf-8") as jf:
        chunks_list = json.load(jf)

    print("Loaded", len(chunks_list), "chunk entries from JSON.")

    # determine embedding dim
    test_vec = encode_texts(["hello world"])
    dim = test_vec.shape[1]

    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)

    metadata = {}
    
    # Dictionary to store stats per chunk_id
    token_stats = {}
    
    next_id = 1

    # Process each chunk entry in the list
    for chunk_entry in tqdm(chunks_list):
        # Each entry is a dict with doc_id as key
        for doc_id, data in chunk_entry.items():
            # Use chunk_text_with_metadata for better retrieval (includes document metadata)
            #text = data["chunk_text_with_metadata"]
            # Alternative: use plain chunk_text
            text = data["chunk_text"]
            span = data["span"]

            # Create a unique chunk_id for this chunk
            chunk_id = f"380_{doc_id}_chunk_{next_id}"

            # 1. Calculate token length without truncation just for logging
            raw_tokens = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
            num_tokens = len(raw_tokens)
            is_truncated = num_tokens > 512

            # 2. Store stats using chunk_id as key
            token_stats[chunk_id] = {
                "token_count": num_tokens,
                "truncated": is_truncated,
                "doc_id": doc_id
            }

            # 3. Create embedding (with safe truncation inside encode_texts)
            emb = encode_texts([text])
            id_arr = np.array([next_id], dtype=np.int64)
            index.add_with_ids(emb, id_arr)

            metadata[str(next_id)] = {
                "chunk_id": chunk_id,
                "span": span,
                "doc_id": doc_id,
                "chunk_text": data["chunk_text"]  # Store original text for reference
            }

            next_id += 1

    # Save FAISS Index
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Save Metadata
    with open(META_PATH, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    # Save Token Stats
    with open(STATS_PATH, "w", encoding="utf-8") as sf:
        json.dump(token_stats, sf, indent=2)

    print("Index saved:", FAISS_INDEX_PATH)
    print("Metadata saved:", META_PATH)
    print("Token stats saved:", STATS_PATH)
    print(f"Total chunks processed: {next_id - 1}")


# =========================
# SEARCH PIPELINE
# =========================
def load_search_components():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    build_faiss_index()