import json
import re
from typing import Dict, Tuple

# ----------------------------------------------------
# Utility functions
# ----------------------------------------------------
def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def span_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Return number of overlapping characters between two spans"""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def normalize_words(text: str):
    """Lowercase, remove non-alphanumerics, split into words"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return text.split()

def text_overlap_retrieved(gt_text: str, chunk_text: str, threshold: float = 0.90):
    """
    Returns (bool, coverage)
    True if at least `threshold` fraction of GT words appear in retrieved chunk
    """
    gt_words = normalize_words(gt_text)
    if len(gt_words) == 0:
        return False, 0.0

    chunk_words = set(normalize_words(chunk_text))
    match_count = sum(1 for w in gt_words if w in chunk_words)
    coverage = match_count / len(gt_words)
    return coverage >= threshold, coverage

# ----------------------------------------------------
# Span-based retrieval evaluation
# ----------------------------------------------------
def evaluate_retrieval(ground_truth: Dict,
                       retrieved: Dict,
                       chunk_store: Dict,
                       top_k: int = 10) -> Dict:

    results = {}

    for qid, q_gt in ground_truth.items():
        if qid not in retrieved:
            results[qid] = {
                "doc_rank_hit": None,
                "span_rank_hit": None,
                "doc_hit": False,
                "span_hit": False
            }
            continue

        gt_file = q_gt["snippets"][0]["file_path"]
        gt_span = tuple(q_gt["snippets"][0]["span"])
        ret = retrieved[qid]["retrieved_chunks"]

        doc_rank_hit = None
        span_rank_hit = None

        for rank in range(1, top_k + 1):
            key = str(rank)
            if key not in ret:
                continue

            r = ret[key]

            if r["retrieved_doc_id"] == gt_file and doc_rank_hit is None:
                doc_rank_hit = rank

            r_span = tuple(r["span"])
            if span_overlap(gt_span, r_span) > 0 and span_rank_hit is None:
                span_rank_hit = rank

        results[qid] = {
            "ground_truth_file": gt_file,
            "gt_span": gt_span,
            "doc_rank_hit": doc_rank_hit,
            "span_rank_hit": span_rank_hit,
            "doc_hit": doc_rank_hit is not None,
            "span_hit": span_rank_hit is not None
        }

    return results

def compute_metrics(results: Dict) -> Dict:
    total = len(results)

    def r_at_k(field, k):
        return sum(1 for r in results.values()
                   if r[field] is not None and r[field] <= k) / total

    metrics = {
        "total_queries": total,
        "doc_hit_rate": sum(1 for r in results.values() if r["doc_hit"]) / total,
        "span_hit_rate": sum(1 for r in results.values() if r["span_hit"]) / total,
        "R1_doc": r_at_k("doc_rank_hit", 1),
        "R5_doc": r_at_k("doc_rank_hit", 5),
        "R10_doc": r_at_k("doc_rank_hit", 10),
        "R1_span": r_at_k("span_rank_hit", 1),
        "R5_span": r_at_k("span_rank_hit", 5),
        "R10_span": r_at_k("span_rank_hit", 10)
    }
    return metrics

# ----------------------------------------------------
# Text-based retrieval evaluation
# ----------------------------------------------------
def evaluate_text_retrieval(ground_truth: Dict,
                            retrieved: Dict,
                            chunk_store: Dict,
                            top_k: int = 10,
                            threshold: float = 0.90) -> Dict:

    results = {}

    for qid, q_gt in ground_truth.items():
        if qid not in retrieved:
            results[qid] = {"text_hit": False, "text_rank_hit": None}
            continue

        gt_text = q_gt["snippets"][0]["answer"]
        ret = retrieved[qid]["retrieved_chunks"]

        text_hit = False
        text_rank_hit = None

        for rank in range(1, top_k + 1):
            key = str(rank)
            if key not in ret:
                continue

            r = ret[key]
            chunk_id = r["chunk_id"]
            if chunk_id not in chunk_store:
                continue

            chunk_text = chunk_store[chunk_id]["chunk_text"]

            ok, coverage = text_overlap_retrieved(gt_text, chunk_text, threshold=threshold)
            if ok and text_rank_hit is None:
                text_rank_hit = rank
                text_hit = True

        results[qid] = {
            "text_hit": text_hit,
            "text_rank_hit": text_rank_hit
        }

    return results

def compute_text_metrics(results: Dict) -> Dict:
    total = len(results)

    def r_at_k(field, k):
        return sum(1 for r in results.values()
                   if r[field] is not None and r[field] <= k) / total

    metrics = {
        "R1_text": r_at_k("text_rank_hit", 1),
        "R5_text": r_at_k("text_rank_hit", 5),
        "R10_text": r_at_k("text_rank_hit", 10),
        "text_hit_rate": sum(1 for r in results.values() if r["text_hit"]) / total
    }
    return metrics

# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == "__main__":
    gt_path = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/meta-enriched-rag-for-legal-llms/RAG_pipeline/embeddings_with_span/privacy_qa_queries_with_ids.json"
    ret_path = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/meta-enriched-rag-for-legal-llms/RAG_pipeline/embeddings_with_span/retrieval_results.json"
    chunks_path = "/home/smaniyar_umass_edu/BioNLP_Ontology/other/meta-enriched-rag-for-legal-llms/RAG_pipeline/embeddings_with_span/privacy_qa_chunks.json"

    gt = load_json(gt_path)
    ret = load_json(ret_path)
    chunks = load_json(chunks_path)

    # Span-based evaluation
    span_results = evaluate_retrieval(gt, ret, chunks, top_k=10)
    span_metrics = compute_metrics(span_results)
    print("=== Span-based Retrieval Metrics ===")
    print(json.dumps(span_metrics, indent=2))

    # Text-based evaluation with 90% threshold
    text_results = evaluate_text_retrieval(gt, ret, chunks, top_k=10, threshold=1)
    text_metrics = compute_text_metrics(text_results)
    print("=== Text-based Retrieval Metrics (100% word coverage) ===")
    print(json.dumps(text_metrics, indent=2))
