import os
import json
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG (edit paths as required)
# ============================================================
INPUT_TEST_JSON = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/Test/privacy_qa.json" # Ground Truth JSON
QUERY_ID_JSON = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/Test/privacy_qa_queries_with_ids.json" #createrd here 
METADATA_JSON = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/embeddings_with_span/metadata_privacy_qa.json" # have from embedding py file
FAISS_INDEX = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/embeddings_with_span/faiss_index_privacy_qa_with_span.bin" #have from embedding py file 
OUTPUT_RETRIEVAL_JSON = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/embeddings_with_span/retrieval_results.json" # Output I get 

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
# STEP 2: Retrieval helpers
# ============================================================
def load_components(faiss_path, meta_path):
    index = faiss.read_index(faiss_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def search_topk(query, index, topk=10):
    q_emb = encode_texts([query])
    scores, ids = index.search(q_emb, topk)
    return scores[0], ids[0]


# ============================================================
# STEP 3: Build output JSON
# ============================================================
def build_output_json(query_id_json, meta_path, faiss_path, output_json):
    index, metadata = load_components(faiss_path, meta_path)

    with open(query_id_json, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    out = {}

    for qid, qinfo in query_data.items():
        query_text = qinfo["query"]
        doc_id = qinfo.get("doc_id")

        scores, ids = search_topk(query_text, index, topk=10)

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


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    assign_query_ids(INPUT_TEST_JSON, QUERY_ID_JSON)

    build_output_json(
        query_id_json=QUERY_ID_JSON,
        meta_path=METADATA_JSON,
        faiss_path=FAISS_INDEX,
        output_json=OUTPUT_RETRIEVAL_JSON
    )
