import os
import json
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import re
import math
from collections import defaultdict, Counter


# ============================================================
# CONFIG (edit paths as required)
# ============================================================
# INPUT_TEST_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa.json" # Ground Truth JSON
QUERY_ID_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_w_query_ids.json" #createrd here OR give path if already generated (dont rerun, comment assign_query_ids call in main)
METADATA_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_embs/faiss_emb_recur_w_keyword_metadata.json" # have from embedding py file
FAISS_INDEX = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_embs/faiss_emb_recur_w_keyword_metadata.bin" #have from embedding py file 
OUTPUT_RETRIEVAL_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_inference/retrieval_results_recur_dense_w_keyword_metadata.json" # Output I get 

EMBED_MODEL = "thenlper/gte-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16
NORMALIZE = True


# ============================================================
# MODEL LOADING
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, use_fast=True)
model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE)
model.eval()


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


def encode_texts(texts):
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model(**inputs)
            emb = mean_pool(output.last_hidden_state, inputs["attention_mask"])

        emb = emb.cpu().numpy().astype("float32")
        out.append(emb)

    embs = np.vstack(out)
    if NORMALIZE:
        faiss.normalize_L2(embs)
    return embs


# ============================================================
# STEP 1: Add query IDs to input JSON + extract doc_id
# ============================================================
def extract_doc_id(snippets):
    if not snippets:
        return None
    fp = snippets[0]["file_path"]
    return fp


def assign_query_ids(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data["tests"]
    out = {}
    qid = 1

    for t in tests:
        key = f"query_id_{qid}"
        doc_id = extract_doc_id(t.get("snippets", []))

        out[key] = {
            "query": t["query"],
            "doc_id": doc_id,
            "snippets": t.get("snippets", [])
        }
        qid += 1

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return output_json_path


# ============================================================
# NEW: Extract retrieved_doc_id from chunk_id
# ============================================================
def extract_doc_id_from_chunkid(chunk_id: str) -> str:
    parts = chunk_id.split("_")
    if len(parts) < 3:
        return None  # malformed
    
    # last part = "chunkN"
    # second last part = filename (without .txt)
    filename = parts[-2] + ".txt"
    
    # everything before last two components is the folder name
    folder = "_".join(parts[:-2])
    
    return f"{folder}/{filename}"



# ============================================================
# STEP 2: Retrieval helpers for Dense Retrieval
# ============================================================
def load_components(faiss_path, meta_path):
    index = faiss.read_index(faiss_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def search_topk(query, index, topk=64):
    q_emb = encode_texts([query])
    scores, ids = index.search(q_emb, topk)
    return scores[0], ids[0]

# ============================================================
# STEP 3: Retrieval helpers for BM25 sparse retrieval
# ============================================================

def _simple_tokenize(text: str):
    """Very simple tokenizer: lowercase + split on non-word chars."""
    tokens = re.split(r"\W+", text.lower())
    return [t for t in tokens if t]


def build_bm25_index(metadata):
    """
    Build a BM25 index over chunks in metadata.

    Expects metadata[cid_str] to contain a "chunk_text" field.
    If it doesn't, you need to add chunk_text to your metadata JSON.
    """
    # Ensure deterministic ordering by numeric FAISS id
    doc_keys = sorted(metadata.keys(), key=lambda x: int(x))

    N = len(doc_keys)
    doc_lens = []
    doc_counters = []
    df = defaultdict(int)
    inverted = defaultdict(list)  # term -> list of (doc_idx, tf)
    meta_key_to_doc_idx = {}

    for doc_idx, key in enumerate(doc_keys):
        meta = metadata[key]
        text = meta.get("chunk_text", "")
        tokens = _simple_tokenize(text)
        doc_len = len(tokens)
        doc_lens.append(doc_len)

        counter = Counter(tokens)
        doc_counters.append(counter)
        meta_key_to_doc_idx[key] = doc_idx

        for term, tf in counter.items():
            df[term] += 1
            inverted[term].append((doc_idx, tf))

    avgdl = sum(doc_lens) / (N or 1)
    idf = {}

    for term, df_t in df.items():
        # BM25 idf with +0.5 smoothing
        idf[term] = math.log(1 + (N - df_t + 0.5) / (df_t + 0.5))

    bm25_index = {
        "doc_keys": doc_keys,              # list of metadata keys (string)
        "doc_lens": doc_lens,              # list of doc lengths
        "avgdl": avgdl,
        "inverted": inverted,              # term -> [(doc_idx, tf), ...]
        "idf": idf,
        "meta_key_to_doc_idx": meta_key_to_doc_idx,
        "k1": 1.5,
        "b": 0.75,
    }
    return bm25_index


def bm25_search(query: str, bm25_index, topk: int = 64):
    """Return topk docs by BM25 score."""
    q_tokens = _simple_tokenize(query)
    if not q_tokens:
        return [], []

    scores = defaultdict(float)
    inverted = bm25_index["inverted"]
    idf = bm25_index["idf"]
    doc_lens = bm25_index["doc_lens"]
    avgdl = bm25_index["avgdl"]
    k1 = bm25_index["k1"]
    b = bm25_index["b"]

    for term in q_tokens:
        if term not in inverted:
            continue
        idf_t = idf.get(term, 0.0)
        for doc_idx, tf in inverted[term]:
            dl = doc_lens[doc_idx]
            denom = tf + k1 * (1.0 - b + b * dl / (avgdl or 1.0))
            score = idf_t * (tf * (k1 + 1.0)) / (denom or 1e-9)
            scores[doc_idx] += score

    if not scores:
        return [], []

    # Sort by score desc, take topk
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    doc_indices = [doc_idx for doc_idx, _ in sorted_docs]
    doc_scores = [score for _, score in sorted_docs]
    return doc_scores, doc_indices


def _min_max_normalize(score_dict):
    """Normalize scores in a dict doc_idx -> score to [0, 1]."""
    if not score_dict:
        return {}
    values = list(score_dict.values())
    s_min = min(values)
    s_max = max(values)
    if s_max == s_min:
        return {k: 1.0 for k in score_dict}
    return {k: (v - s_min) / (s_max - s_min) for k, v in score_dict.items()}

def combined_dense_sparse_search(
    query: str,
    index,                # FAISS index
    metadata: dict,
    bm25_index: dict,
    topk: int = 64,
    alpha: float = 0.5    # weight for dense; (1-alpha) for sparse
):
    """
    Perform dense (FAISS) + sparse (BM25) retrieval and fuse scores.

    - Dense scores are cosine similarities from FAISS
    - Sparse scores are BM25
    - Both are min-max normalized, then combined:
        combined = alpha * dense_norm + (1 - alpha) * sparse_norm
    """
    # Dense
    dense_scores_arr, dense_ids_arr = search_topk(query, index, topk=topk)

    dense_scores = {}
    for sc, cid in zip(dense_scores_arr, dense_ids_arr):
        if cid == -1:
            continue
        meta_key = str(int(cid))
        if meta_key not in metadata:
            continue
        # Map meta_key -> doc_idx used by BM25
        doc_idx = bm25_index["meta_key_to_doc_idx"].get(meta_key)
        if doc_idx is None:
            continue
        dense_scores[doc_idx] = float(sc)

    # Sparse (BM25)
    bm25_scores_list, bm25_doc_indices = bm25_search(query, bm25_index, topk=topk)

    sparse_scores = {}
    for sc, doc_idx in zip(bm25_scores_list, bm25_doc_indices):
        sparse_scores[doc_idx] = float(sc)

    # Normalize each score set
    dense_norm = _min_max_normalize(dense_scores)
    sparse_norm = _min_max_normalize(sparse_scores)

    # Combine
    combined_scores = {}
    all_doc_indices = set(dense_norm.keys()) | set(sparse_norm.keys())
    for doc_idx in all_doc_indices:
        d = dense_norm.get(doc_idx, 0.0)
        s = sparse_norm.get(doc_idx, 0.0)
        combined_scores[doc_idx] = alpha * d + (1.0 - alpha) * s

    # Sort and keep topk
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:topk]

    # Map back to metadata keys and original FAISS ids
    doc_keys = bm25_index["doc_keys"]
    results = []
    for doc_idx, score in sorted_docs:
        meta_key = doc_keys[doc_idx]  # this is the metadata key (string)
        meta = metadata.get(meta_key)
        if not meta:
            continue
        # meta["chunk_id"] already has the human-readable id we use everywhere
        results.append((score, meta))

    return results


# ============================================================
# STEP 4: Build output JSON
# ============================================================
def build_output_json(query_id_json, meta_path, faiss_path, output_json):
    index, metadata = load_components(faiss_path, meta_path)

    with open(query_id_json, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    out = {}

    for qid, qinfo in query_data.items():
        query_text = qinfo["query"]
        doc_id = qinfo.get("doc_id")

        scores, ids = search_topk(query_text, index, topk=64)

        retrieved = {}
        rank = 1

        for sc, cid in zip(scores, ids):
            if cid == -1:
                continue

            cid = int(cid)
            cid_str = str(cid)

            if cid_str not in metadata:
                continue

            meta = metadata[cid_str]

            retrieved_doc_id = extract_doc_id_from_chunkid(meta["chunk_id"])

            retrieved[str(rank)] = {
                "score": float(sc),
                "chunk_id": meta["chunk_id"],
                "retrieved_doc_id": retrieved_doc_id,
                "span": meta["span"]
            }

            rank += 1

        out[qid] = {
            "query": query_text,
            "doc_id": doc_id,
            "retrieved_chunks": retrieved
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return output_json

def build_output_json_dense_sparse(
    query_id_json,
    meta_path,
    faiss_path,
    output_json,
    topk=64,
    alpha=0.5
):
    """
    Build retrieval JSON using dense + sparse fusion.

    - Dense: FAISS over GTE embeddings
    - Sparse: BM25 over chunk_text
    - Fusion: combined = alpha * dense_norm + (1-alpha) * sparse_norm
    """
    index, metadata = load_components(faiss_path, meta_path)

    # Build BM25 index once
    bm25_index = build_bm25_index(metadata)

    with open(query_id_json, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    out = {}

    for qid, qinfo in query_data.items():
        query_text = qinfo["query"]
        doc_id = qinfo.get("doc_id")

        fused_results = combined_dense_sparse_search(
            query_text,
            index=index,
            metadata=metadata,
            bm25_index=bm25_index,
            topk=topk,
            alpha=alpha
        )

        retrieved = {}
        rank = 1

        for sc, meta in fused_results:
            retrieved_doc_id = extract_doc_id_from_chunkid(meta["chunk_id"])

            retrieved[str(rank)] = {
                "score": float(sc),
                "chunk_id": meta["chunk_id"],
                "retrieved_doc_id": retrieved_doc_id,
                "span": meta["span"]
            }
            rank += 1

        out[qid] = {
            "query": query_text,
            "doc_id": doc_id,
            "retrieved_chunks": retrieved
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return output_json

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    # assign_query_ids(INPUT_TEST_JSON, QUERY_ID_JSON)

    # build_output_json(
    #     query_id_json=QUERY_ID_JSON,
    #     meta_path=METADATA_JSON,
    #     faiss_path=FAISS_INDEX,
    #     output_json=OUTPUT_RETRIEVAL_JSON
    # )

    build_output_json_dense_sparse(
    query_id_json=QUERY_ID_JSON,
    meta_path=METADATA_JSON,
    faiss_path=FAISS_INDEX,
    output_json=OUTPUT_RETRIEVAL_JSON,
    topk=64,
    alpha=0.8  # tune fusion weight if needed
)
