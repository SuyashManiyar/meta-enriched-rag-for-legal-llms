import json
from pathlib import Path
import re
# ============================================================
# INPUT PATHS
# ============================================================

GROUND_TRUTH_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_recur_data/privacy_qa_queries_with_ids_recur_bm25.json"
RETRIEVAL_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_recur_data/retrieval_results_recur_bm25.json" 
CHUNK_TEXT_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_recur_data/privacy_qa_chunks_recur.json"
OUTPUT_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_recur_data/exp2_recursive_dense_bm25_cosine.json"

# ============================================================

K_VALUES = [1, 2, 4, 8,16,32,64]







# ============================================================
# UTILITIES
# ============================================================

def load_json(path):
    return json.loads(Path(path).read_text())


def spans_overlap(span1, span2):
    a1, a2 = span1
    b1, b2 = span2
    return not (a2 < b1 or b2 < a1)


def tokenize_words(text):
    return re.findall(r"\w+", text.lower())


def word_overlap_sim(gt_text, chunk_text):
    gt_words = tokenize_words(gt_text)
    chunk_words = tokenize_words(chunk_text)

    if len(gt_words) == 0:
        return 0.0

    gt_set = set(gt_words)
    chunk_set = set(chunk_words)

    overlap = len(gt_set.intersection(chunk_set)) / len(gt_set)
    return overlap


# ============================================================
# GROUND TRUTH INDEX
# ============================================================

def index_ground_truth(gt_raw):
    lookup = {}
    if isinstance(gt_raw, list):
        items = gt_raw
    elif isinstance(gt_raw, dict):
        items = gt_raw.values()
    else:
        raise ValueError("Ground truth JSON must be list or dict")

    for item in items:
        q = item["query"]
        lookup[q] = [
            {"file_path": s["file_path"], "span": s["span"], "answer": s.get("answer", "")}
            for s in item["snippets"]
        ]
    return lookup


# ============================================================
# EVALUATION FOR A SINGLE QUERY
# ============================================================

def evaluate_query(qtext, gt_snippets, retrieved_chunks, chunk_text_map):
    # ------- Document Retrieval correctness -------
    gt_docs = set([s["file_path"] for s in gt_snippets])

    # ------- Standard span-based correctness -------
    span_correct_flags = []

    # ------- Word-overlap correctness -------
    word_correct_flags = []

    # Convert GT answer-texts to word lists for text-based eval
    gt_answer_texts = [s["answer"] for s in gt_snippets]

    # For ordering retrieval by rank
    sorted_ranks = sorted(retrieved_chunks, key=lambda x: int(x))

    for rank in sorted_ranks:
        r = retrieved_chunks[rank]
        r_doc = r["retrieved_doc_id"]
        r_span = r["span"]
        r_chunk_id = r["chunk_id"]

        # ----------------------------
        # Document retrieval correctness (binary)
        # ----------------------------
        doc_match = (r_doc in gt_docs)

        # ----------------------------
        # Span-overlap-based correctness
        # ----------------------------
        span_match = False
        if doc_match:
            for s in gt_snippets:
                if s["file_path"] == r_doc and spans_overlap(r_span, s["span"]):
                    span_match = True
                    break
        span_correct_flags.append(span_match)

        # ----------------------------
        # Word-overlap-based correctness
        # ----------------------------
        chunk_text = chunk_text_map[r_chunk_id]["chunk_text"]
        alt_match = False
        for gt_text in gt_answer_texts:
            sim = word_overlap_sim(gt_text, chunk_text)
            if sim >= 0.90:
                alt_match = True
                break
        word_correct_flags.append(alt_match)

    # ===== Compute metrics for k =====

    results = {
        "doc_retrieved_at_k": {},
        "span_precision": {},
        "span_recall": {},
        "alt_precision": {},
        "alt_recall": {}
    }

    total_gt = len(gt_snippets)

    for k in K_VALUES:
        topk_span = span_correct_flags[:k]
        topk_word = word_correct_flags[:k]

        span_correct = sum(topk_span)
        word_correct = sum(topk_word)

        # Document retrieval success if ANY top-k retrieval hits the correct document
        doc_success = any(retrieved_chunks[str(i+1)]["retrieved_doc_id"] in gt_docs
                          for i in range(min(k, len(sorted_ranks))))

        results["doc_retrieved_at_k"][k] = 1 if doc_success else 0

        results["span_precision"][k] = span_correct / k
        results["span_recall"][k] = span_correct / total_gt

        results["alt_precision"][k] = word_correct / k
        results["alt_recall"][k] = word_correct / total_gt

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    gt_raw = load_json(GROUND_TRUTH_JSON)
    ret_raw = load_json(RETRIEVAL_JSON)
    chunk_text_map = load_json(CHUNK_TEXT_JSON)

    gt_index = index_ground_truth(gt_raw)

    per_query_results = {}
    accum_doc = {k: [] for k in K_VALUES}
    accum_span_p = {k: [] for k in K_VALUES}
    accum_span_r = {k: [] for k in K_VALUES}
    accum_alt_p = {k: [] for k in K_VALUES}
    accum_alt_r = {k: [] for k in K_VALUES}

    for qid, entry in ret_raw.items():
        q = entry["query"]
        if q not in gt_index:
            continue

        metrics = evaluate_query(
            q,
            gt_index[q],
            entry["retrieved_chunks"],
            chunk_text_map
        )
        per_query_results[qid] = {"query": q, "metrics": metrics}

        for k in K_VALUES:
            accum_doc[k].append(metrics["doc_retrieved_at_k"][k])
            accum_span_p[k].append(metrics["span_precision"][k])
            accum_span_r[k].append(metrics["span_recall"][k])
            accum_alt_p[k].append(metrics["alt_precision"][k])
            accum_alt_r[k].append(metrics["alt_recall"][k])

    macro_results = {
        "doc_retrieved_macro": {k: sum(accum_doc[k]) / len(accum_doc[k]) for k in K_VALUES},
        "span_precision_macro": {k: sum(accum_span_p[k]) / len(accum_span_p[k]) for k in K_VALUES},
        "span_recall_macro": {k: sum(accum_span_r[k]) / len(accum_span_r[k]) for k in K_VALUES},
        "alt_precision_macro": {k: sum(accum_alt_p[k]) / len(accum_alt_p[k]) for k in K_VALUES},
        "alt_recall_macro": {k: sum(accum_alt_r[k]) / len(accum_alt_r[k]) for k in K_VALUES},
    }

    output = {
        "per_query": per_query_results,
        "macro": macro_results,
        "k_values": K_VALUES
    }

    Path(OUTPUT_JSON).write_text(json.dumps(output, indent=2))
    print(f"Saved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
