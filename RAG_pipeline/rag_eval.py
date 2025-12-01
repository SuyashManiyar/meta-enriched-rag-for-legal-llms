import json
from typing import Dict, Tuple

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

# ----------------------------------------------------
# Span overlap
# ----------------------------------------------------
def span_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

# ----------------------------------------------------
# Core evaluator
# ----------------------------------------------------
def evaluate_retrieval(ground_truth: Dict,
                       retrieved: Dict,
                       chunk_store: Dict,
                       top_k: int = 10) -> Dict:

    results = {}

    for qid, q_gt in ground_truth.items():
        if qid not in retrieved:
            results[qid] = {
                "ground_truth_file": None,
                "gt_span": None,
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

# ----------------------------------------------------
# Aggregate metrics: overall + R@K
# ----------------------------------------------------
def compute_metrics(results: Dict) -> Dict:
    total = len(results)

    doc_hits = sum(1 for r in results.values() if r["doc_hit"])
    span_hits = sum(1 for r in results.values() if r["span_hit"])

    def r_at_k(field: str, k: int) -> float:
        return sum(
            1 for r in results.values()
            if r[field] is not None and r[field] <= k
        ) / total

    metrics = {
        "total_queries": total,

        "doc_hit_rate": doc_hits / total,
        "span_hit_rate": span_hits / total,

        "R1_doc": r_at_k("doc_rank_hit", 1),
        "R5_doc": r_at_k("doc_rank_hit", 5),
        "R10_doc": r_at_k("doc_rank_hit", 10),

        "R1_span": r_at_k("span_rank_hit", 1),
        "R5_span": r_at_k("span_rank_hit", 5),
        "R10_span": r_at_k("span_rank_hit", 10)
    }

    return metrics

# ----------------------------------------------------
# Usage Example (your exact paths)
# ----------------------------------------------------
if __name__ == "__main__":
    gt = load_json(
        "/home/smaniyar_umass_edu/BioNLP_Ontology/other/meta-enriched-rag-for-legal-llms/"
        "RAG_pipeline/embeddings_with_span/privacy_qa_queries_with_ids.json"
    )

    ret = load_json(
        "/home/smaniyar_umass_edu/BioNLP_Ontology/other/meta-enriched-rag-for-legal-llms/"
        "RAG_pipeline/embeddings_with_span/retrieval_results.json"
    )

    chunks = load_json(
        "/home/smaniyar_umass_edu/BioNLP_Ontology/other/meta-enriched-rag-for-legal-llms/"
        "RAG_pipeline/embeddings_with_span/privacy_qa_chunks.json"
    )

    results = evaluate_retrieval(gt, ret, chunks)
    metrics = compute_metrics(results)

    print(json.dumps(metrics, indent=2))
    # print(json.dumps(results, indent=2))
