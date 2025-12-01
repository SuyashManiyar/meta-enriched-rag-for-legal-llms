import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

# =========================
# CONFIG
# =========================
EMBED_MODEL = "thenlper/gte-large"
CHUNK_JSON_PATH = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/privacy_qa_chunks.json"

FAISS_INDEX_PATH = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/embeddings_with_span/faiss_index_privacy_qa_with_span.bin"
META_PATH = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/embeddings_with_span/metadata_privacy_qa.json"

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
        inputs = tokenizer(b, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
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
    # Load chunk dictionary
    with open(CHUNK_JSON_PATH, "r", encoding="utf-8") as jf:
        chunks = json.load(jf)

    print("Loaded", len(chunks), "chunks from JSON.")

    # determine embedding dim
    test_vec = encode_texts(["hello world"])
    dim = test_vec.shape[1]

    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)

    metadata = {}
    next_id = 1

    for chunk_id, data in chunks.items():
        text = data["chunk_text"]
        span = data["span"]

        emb = encode_texts([text])
        id_arr = np.array([next_id], dtype=np.int64)
        index.add_with_ids(emb, id_arr)

        metadata[str(next_id)] = {
            "chunk_id": chunk_id,
            "span": span
        }

        next_id += 1

    # save
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    print("Index saved:", FAISS_INDEX_PATH)
    print("Metadata saved:", META_PATH)


# =========================
# SEARCH PIPELINE
# =========================
def load_search_components():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search(query, topk=5):
    index, metadata = load_search_components()

    q_emb = encode_texts([query])
    scores, ids = index.search(q_emb, topk)

    results = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        meta = metadata[str(int(i))]
        results.append({
            "id": int(i),
            "score": float(s),
            "chunk_id": meta["chunk_id"],
            "span": meta["span"]
        })
    return results


# =========================
# MAIN
# =========================
build_faiss_index()
