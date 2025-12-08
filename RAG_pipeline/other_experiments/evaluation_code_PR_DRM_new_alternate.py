import json
from pathlib import Path
import re
from tqdm import tqdm 
# ============================================================
# INPUT PATHS
# ============================================================

GROUND_TRUTH_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_w_query_ids.json"
RETRIEVAL_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_inference/retrieval_results_recur_dense_window_metadata_n_doc_name_corrected.json"
CHUNK_TEXT_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_chunks/chunks_window_summary_n_doc_name.json"
OUTPUT_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_evals/eval_recur_dense_w_window_metadata_n_doc_name.json"

# GROUND_TRUTH_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_w_query_ids.json"
# RETRIEVAL_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_inference/retrieval_results_recur_dense_w_window_metadata_n_doc_name.json"
# CHUNK_TEXT_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_chunks/chunks_window_summary_n_doc_name.json"
# OUTPUT_JSON = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_evals/eval_recur_dense_w_window_metadata_n_doc_name.json"
# ============================================================

K_VALUES = [1, 2, 4, 8, 16, 32, 64]

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

def word_overlap_sim(gt_text, chunk_text, threshold=0.75):
    """
    Returns 1.0 if there is a contiguous match between gt_text tokens
    and chunk_text tokens with fraction >= threshold.
    Only considers:
      1. GT fully inside chunk
      2. Ending of GT at start of chunk
      3. Start of GT at end of chunk
    """
    gt_tokens = tokenize_words(gt_text.lower())
    chunk_tokens = tokenize_words(chunk_text.lower())

    if not gt_tokens or not chunk_tokens:
        return 0.0

    gt_len = len(gt_tokens)
    required = int(threshold * gt_len)  # minimum contiguous tokens

    # Case 1: GT fully inside chunk
    for i in range(len(chunk_tokens) - gt_len + 1):
        if chunk_tokens[i:i + gt_len] == gt_tokens:
            return 1.0

    # Case 2: Ending of GT at start of chunk
    max_possible = min(len(chunk_tokens), gt_len)
    for k in range(max_possible, 0, -1):
        if chunk_tokens[:k] == gt_tokens[-k:] and k >= required:
            return 1.0

    # Case 3: Start of GT at end of chunk
    for k in range(max_possible, 0, -1):
        if chunk_tokens[-k:] == gt_tokens[:k] and k >= required:
            return 1.0

    return 0.0


# def word_overlap_sim(gt_text, chunk_text, threshold=0.9):
#     """
#     Returns 1.0 if ANY contiguous substring of GT tokens
#     appears in the chunk tokens in the same order, and the
#     fraction of GT tokens matched >= threshold.
#     """

#     # Normalize + tokenize
#     gt_tokens = tokenize_words(gt_text.lower())
#     chunk_tokens = tokenize_words(chunk_text.lower())

#     if not gt_tokens or not chunk_tokens:
#         return 0.0

#     gt_len = len(gt_tokens)
#     max_match = 0

#     # Generate all contiguous substrings of gt_tokens
#     for start in range(gt_len):
#         for end in range(start + 1, gt_len + 1):  # end is exclusive
#             sub_gt = gt_tokens[start:end]
#             sub_len = len(sub_gt)

#             # Slide sub_gt over chunk_tokens
#             for i in range(len(chunk_tokens) - sub_len + 1):
#                 if chunk_tokens[i:i + sub_len] == sub_gt:
#                     max_match = max(max_match, sub_len)

#     overlap_ratio = max_match / gt_len
    return 1.0 if overlap_ratio >= threshold else 0.0


# def word_overlap_sim(gt_text, chunk_text):
#     gt_words = tokenize_words(gt_text)
#     chunk_words = tokenize_words(chunk_text)

#     if len(gt_words) == 0:
#         return 0.0

#     gt_set = set(gt_words)
#     chunk_set = set(chunk_words)

#     overlap = len(gt_set.intersection(chunk_set)) / len(gt_set)
#     return overlap

# def word_overlap_sim(gt_text, chunk_text):
#     """
#     Returns 1.0 only if the ground-truth text (lowercased, normalized)
#     appears as a contiguous substring inside the retrieved chunk text
#     (also lowercased and normalized).
#     """

#     # Lowercase
#     gt = gt_text.lower()
#     chunk = chunk_text.lower()

#     # Normalize whitespace (removes repeated spaces, newlines, tabs)
#     gt = " ".join(gt.split())
#     chunk = " ".join(chunk.split())

#     # If GT is empty, treat as no-match
#     if len(gt.strip()) == 0:
#         return 0.0

#     # Check exact contiguous substring match
#     return 1.0 if gt in chunk else 0.0
# def word_overlap_sim(gt_text, chunk_text, threshold=0.9):
#     """
#     Check if at least threshold% of GT tokens appear in-order
#     inside the chunk tokens. Case-insensitive. Allows gaps.
#     """

#     # Normalize
#     gt_tokens = tokenize_words(gt_text.lower())
#     chunk_tokens = tokenize_words(chunk_text.lower())

#     if not gt_tokens:
#         return 0.0

#     gt_len = len(gt_tokens)
#     required = int(threshold * gt_len)

#     # in-order subsequence scan
#     i = 0  # index in GT
#     for tok in chunk_tokens:
#         if i < gt_len and tok == gt_tokens[i]:
#             i += 1
#         if i >= required:
#             return 1.0  # enough matched in order

#     return 0.0


# def word_overlap_sim(gt_text, chunk_text, threshold=0.90):
#     """
#     Returns 1.0 if ANY contiguous substring in the chunk
#     overlaps GT tokens in the same order by >= threshold.
#     """

#     # Normalize + tokenize
#     gt_tokens = tokenize_words(gt_text.lower())
#     chunk_tokens = tokenize_words(chunk_text.lower())
#     print(gt_tokens)

#     if not gt_tokens:
#         return 0.0

#     gt_len = len(gt_tokens)
#     if gt_len == 0:
#         return 0.0

#     # Slide a window over chunk tokens
#     max_match = 0

#     for i in range(len(chunk_tokens)):
#         match_count = 0

#         # Compare GT tokens to chunk starting at i
#         for j in range(gt_len):
#             if i + j >= len(chunk_tokens):
#                 break
#             if chunk_tokens[i + j] == gt_tokens[j]:
#                 match_count += 1
#             else:
#                 break  # must be contiguous exact match

#         max_match = max(max_match, match_count)


#     # compute ratio
#     overlap_ratio = max_match / gt_len
#     return 1.0 if overlap_ratio >= threshold else 0.0


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
# EVALUATION FOR A SINGLE QUERY (with DRM + cases)
# ============================================================

def evaluate_query(qtext, gt_snippets, retrieved_chunks, chunk_text_map):
    gt_docs = set([s["file_path"] for s in gt_snippets])

    span_correct_flags = []
    word_correct_flags = []
    drm_flags = []  # wrong document = 1

    gt_answer_texts = [s["answer"] for s in gt_snippets]
    sorted_ranks = sorted(retrieved_chunks, key=lambda x: int(x))

    for rank in sorted_ranks:
        r = retrieved_chunks[rank]
        r_doc = r["retrieved_doc_id"]
        r_span = r["span"]
        r_chunk_id = r["chunk_id"]

        # ---- DRM ----
        doc_match = (r_doc in gt_docs)
        drm_flags.append(0 if doc_match else 1)

        # ---- Span correctness ----
        span_match = False
        if doc_match:
            for s in gt_snippets:
                if s["file_path"] == r_doc and spans_overlap(r_span, s["span"]):
                    span_match = True
                    break
        span_correct_flags.append(span_match)

        # ---- Alt word correctness ----
        chunk_text = chunk_text_map[r_chunk_id]["chunk_text"]
        alt_match = False
        for gt_text in gt_answer_texts:
            if word_overlap_sim(gt_text, chunk_text) == 1:
                alt_match = True
                break
        word_correct_flags.append(alt_match)

    # ============================================================
    # METRICS FOR THIS QUERY
    # ============================================================

    results = {
        "doc_retrieved_at_k": {},
        "span_precision": {},
        "span_recall": {},
        "alt_precision": {},
        "alt_recall": {},
        "drm": {}  # NEW
    }

    total_gt = len(gt_snippets)

    for k in K_VALUES:
        topk_span = span_correct_flags[:k]
        topk_word = word_correct_flags[:k]
        topk_drm = drm_flags[:k]

        span_correct = sum(topk_span)
        word_correct = sum(topk_word)

        # doc success if ANY correct doc in top-k
        doc_success = any(
            retrieved_chunks[str(i+1)]["retrieved_doc_id"] in gt_docs
            for i in range(min(k, len(sorted_ranks)))
        )

        # store metrics
        results["doc_retrieved_at_k"][k] = 1 if doc_success else 0
        results["span_precision"][k] = span_correct / k
        results["span_recall"][k] = span_correct / total_gt
        results["alt_precision"][k] = word_correct / k
        results["alt_recall"][k] = word_correct / total_gt
        results["drm"][k] = sum(topk_drm) / k  # DRM

    return results, span_correct_flags, word_correct_flags

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
    accum_drm = {k: [] for k in K_VALUES}  # NEW

    # correctness-case buckets
    case_span_correct_alt_wrong = []
    case_span_wrong_alt_correct = []
    case_both_correct = []
    case_both_wrong = []

    # evaluate each query
    for qid, entry in tqdm(ret_raw.items()):
        q = entry["query"]
        if q not in gt_index:
            continue

        metrics, span_flags, alt_flags = evaluate_query(
            q,
            gt_index[q],
            entry["retrieved_chunks"],
            chunk_text_map
        )

        per_query_results[qid] = {"query": q, "metrics": metrics}

        # correctness-case classification (top-1)
        # correctness-case classification at R@16
        # K = 16

        # span_k = any(span_flags[:K])
        # alt_k = any(alt_flags[:K])

        # if span_k and not alt_k:
        #     case_span_correct_alt_wrong.append(qid)
        # elif (not span_k) and alt_k:
        #     case_span_wrong_alt_correct.append(qid)
        # elif span_k and alt_k:
        #     case_both_correct.append(qid)
        # else:
        #     case_both_wrong.append(qid)
        # correctness-case classification (top-1)
        span1 = span_flags[0]
        alt1 = alt_flags[0]

        if span1 and not alt1:
            case_span_correct_alt_wrong.append(qid)
        elif (not span1) and alt1:
            case_span_wrong_alt_correct.append(qid)
        elif span1 and alt1:
            case_both_correct.append(qid)
        else:
            case_both_wrong.append(qid)


        # accumulate macro metrics
        for k in K_VALUES:
            accum_doc[k].append(metrics["doc_retrieved_at_k"][k])
            accum_span_p[k].append(metrics["span_precision"][k])
            accum_span_r[k].append(metrics["span_recall"][k])
            accum_alt_p[k].append(metrics["alt_precision"][k])
            accum_alt_r[k].append(metrics["alt_recall"][k])
            accum_drm[k].append(metrics["drm"][k])

    # ============================================================
    # MACRO RESULTS
    # ============================================================

    macro_results = {
        "doc_retrieved_macro": {k: sum(accum_doc[k]) / len(accum_doc[k]) for k in K_VALUES},
        "span_precision_macro": {k: sum(accum_span_p[k]) / len(accum_span_p[k]) for k in K_VALUES},
        "span_recall_macro": {k: sum(accum_span_r[k]) / len(accum_span_r[k]) for k in K_VALUES},
        "alt_precision_macro": {k: sum(accum_alt_p[k]) / len(accum_alt_p[k]) for k in K_VALUES},
        "alt_recall_macro": {k: sum(accum_alt_r[k]) / len(accum_alt_r[k]) for k in K_VALUES},
        "drm_macro": {k: sum(accum_drm[k]) / len(accum_drm[k]) for k in K_VALUES},
    }

    # ============================================================
    # CASE PERCENTAGES
    # ============================================================

    total_queries = (
        len(case_span_correct_alt_wrong) +
        len(case_span_wrong_alt_correct) +
        len(case_both_correct) +
        len(case_both_wrong)
    )

    def pct(x):
        return (x / total_queries) * 100 if total_queries > 0 else 0.0

    case_stats = {
        "span_correct_alt_wrong": {
            "query_ids": case_span_correct_alt_wrong,
            "count": len(case_span_correct_alt_wrong),
            "percentage": pct(len(case_span_correct_alt_wrong))
        },
        "span_wrong_alt_correct": {
            "query_ids": case_span_wrong_alt_correct,
            "count": len(case_span_wrong_alt_correct),
            "percentage": pct(len(case_span_wrong_alt_correct))
        },
        "both_correct": {
            "query_ids": case_both_correct,
            "count": len(case_both_correct),
            "percentage": pct(len(case_both_correct))
        },
        "both_wrong": {
            "query_ids": case_both_wrong,
            "count": len(case_both_wrong),
            "percentage": pct(len(case_both_wrong))
        },
        "total_queries_evaluated": total_queries
    }

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    output = {
        "per_query": per_query_results,
        "macro": macro_results,
        "k_values": K_VALUES,
        "cases": case_stats
    }

    Path(OUTPUT_JSON).write_text(json.dumps(output, indent=2))
    print(f"Saved results to {OUTPUT_JSON}")


# ============================================================
# EXECUTE MAIN
# ============================================================

if __name__ == "__main__":
    main()
