import json
from pathlib import Path
import re
import os
from tqdm import tqdm

# ============================================================
# EVALUATION CONFIGURATIONS
# ============================================================

EVALUATIONS = [
    # Original evaluations
    {
        "name": "recursive_dense_sparse",
        "ground_truth": "australian_legal_data/RAG_ground_truth_corrected.json",
        "retrieval_results": "australian_legal_data/results_recursive/australian_legal_retrieval_results_recursive_dense_sparse.json",
        "chunk_text": "australian_legal_data/generated_chunks/australian_legal_text_recursive_chunking.json",
        "output": "australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_sparse_results.json"
    },
    {
        "name": "recursive_dense_only",
        "ground_truth": "australian_legal_data/RAG_ground_truth_corrected.json",
        "retrieval_results": "australian_legal_data/results_recursive/australian_legal_retrieval_results_recursive_dense_only.json",
        "chunk_text": "australian_legal_data/generated_chunks/australian_legal_text_recursive_chunking.json",
        "output": "australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_only_results.json"
    },
    {
        "name": "meta_recursive_dense_sparse",
        "ground_truth": "australian_legal_data/RAG_ground_truth_corrected.json",
        "retrieval_results": "australian_legal_data/results_meta_recursive/australian_legal_retrieval_results_meta_recursive_dense_sparse.json",
        "chunk_text": "australian_legal_data/generated_chunks/australian_legal_text_meta_recursive_chunking.json",
        "output": "australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_sparse_results.json"
    },
    {
        "name": "meta_recursive_dense_only",
        "ground_truth": "australian_legal_data/RAG_ground_truth_corrected.json",
        "retrieval_results": "australian_legal_data/results_meta_recursive/australian_legal_retrieval_results_meta_recursive_dense_only.json",
        "chunk_text": "australian_legal_data/generated_chunks/australian_legal_text_meta_recursive_chunking.json",
        "output": "australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_only_results.json"
    },
    
    # Alternate evaluations with DRM and case analysis
    {
        "name": "recursive_dense_sparse_alternate",
        "ground_truth": "australian_legal_data/RAG_ground_truth_corrected.json",
        "retrieval_results": "australian_legal_data/results_recursive/australian_legal_retrieval_results_recursive_dense_sparse.json",
        "chunk_text": "australian_legal_data/generated_chunks/australian_legal_text_recursive_chunking.json",
        "output": "australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_sparse_alternate_results.json",
        "use_alternate": True
    },
    {
        "name": "recursive_dense_only_alternate",
        "ground_truth": "australian_legal_data/RAG_ground_truth_corrected.json",
        "retrieval_results": "australian_legal_data/results_recursive/australian_legal_retrieval_results_recursive_dense_only.json",
        "chunk_text": "australian_legal_data/generated_chunks/australian_legal_text_recursive_chunking.json",
        "output": "australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_only_alternate_results.json",
        "use_alternate": True
    },
    {
        "name": "meta_recursive_dense_sparse_alternate",
        "ground_truth": "australian_legal_data/RAG_ground_truth_corrected.json",
        "retrieval_results": "australian_legal_data/results_meta_recursive/australian_legal_retrieval_results_meta_recursive_dense_sparse.json",
        "chunk_text": "australian_legal_data/generated_chunks/australian_legal_text_meta_recursive_chunking.json",
        "output": "australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_sparse_alternate_results.json",
        "use_alternate": True
    },
    {
        "name": "meta_recursive_dense_only_alternate",
        "ground_truth": "australian_legal_data/RAG_ground_truth_corrected.json",
        "retrieval_results": "australian_legal_data/results_meta_recursive/australian_legal_retrieval_results_meta_recursive_dense_only.json",
        "chunk_text": "australian_legal_data/generated_chunks/australian_legal_text_meta_recursive_chunking.json",
        "output": "australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_only_alternate_results.json",
        "use_alternate": True
    }
]

K_VALUES = [1, 2, 4, 8, 16, 32, 64]

# ============================================================
# UTILITIES
# ============================================================

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

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

def word_overlap_sim_original(gt_text, chunk_text):
    """Original word overlap similarity for backward compatibility"""
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
    """
    Index ground truth for Australian legal data format.
    Expected format: {qa_id: {question, citation, text, span, document_path}}
    """
    lookup = {}
    
    if isinstance(gt_raw, dict):
        # Australian legal format: direct dict of QA pairs
        for qa_id, qa_data in gt_raw.items():
            question = qa_data.get("question", "")
            text = qa_data.get("text", "")  # Use text instead of answer for RAG eval
            span = qa_data.get("span", [])
            doc_path = qa_data.get("document_path", "")
            
            # Extract document ID from QA ID since document_path might be null
            # QA ID format: "001", "003", "005", etc.
            doc_id = qa_id
            
            # If document_path is available, try to extract from it as well
            if doc_path and doc_path != "null":
                import re
                match = re.search(r'\\(\d+)_', doc_path) or re.search(r'/(\d+)_', doc_path)
                if match:
                    doc_id = match.group(1)
            
            lookup[question] = [{
                "file_path": doc_id,
                "span": span if span and span != "null" else [],
                "text": text,  # Use text field
                "document_path": doc_path,
                "qa_id": qa_id  # Keep original QA ID for matching
            }]
    
    return lookup

# ============================================================
# EVALUATION FOR A SINGLE QUERY
# ============================================================

def evaluate_query(qtext, gt_snippets, retrieved_chunks, chunk_text_map, use_alternate=False):
    """
    Evaluate a single query with both original and alternate approaches.
    
    Args:
        use_alternate: If True, uses the alternate evaluation approach with DRM and case analysis
    """
    # ------- Document Retrieval correctness -------
    gt_docs = set([s["file_path"] for s in gt_snippets])

    # ------- Standard span-based correctness -------
    span_correct_flags = []

    # ------- Word-overlap correctness -------
    word_correct_flags = []
    
    # ------- DRM flags (for alternate approach) -------
    drm_flags = []  # wrong document = 1

    # Convert GT texts to word lists for text-based eval
    gt_texts = [s["text"] for s in gt_snippets]

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
        
        # DRM flag (for alternate approach)
        drm_flags.append(0 if doc_match else 1)

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
        alt_match = False
        if r_chunk_id in chunk_text_map:
            chunk_text = chunk_text_map[r_chunk_id]["chunk_text"]
            for gt_text in gt_texts:
                if use_alternate:
                    # Alternate approach: strict contiguous matching
                    if word_overlap_sim(gt_text, chunk_text) == 1.0:
                        alt_match = True
                        break
                else:
                    # Original approach: threshold-based matching
                    sim = word_overlap_sim_original(gt_text, chunk_text)
                    if sim >= 0.60:
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
    
    # Add DRM for alternate approach
    if use_alternate:
        results["drm"] = {}

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

        results["span_precision"][k] = span_correct / k if k > 0 else 0
        results["span_recall"][k] = span_correct / total_gt if total_gt > 0 else 0

        results["alt_precision"][k] = word_correct / k if k > 0 else 0
        results["alt_recall"][k] = word_correct / total_gt if total_gt > 0 else 0
        
        # Add DRM metric for alternate approach
        if use_alternate:
            topk_drm = drm_flags[:k]
            results["drm"][k] = sum(topk_drm) / k if k > 0 else 0

    if use_alternate:
        return results, span_correct_flags, word_correct_flags
    else:
        return results

# ============================================================
# SINGLE EVALUATION FUNCTION
# ============================================================

def run_single_evaluation(config):
    """Run evaluation for a single configuration"""
    print(f"\n=== Running evaluation: {config['name']} ===")
    
    # Check if files exist
    for key, path in config.items():
        if key in ['ground_truth', 'retrieval_results', 'chunk_text']:
            if not Path(path).exists():
                print(f"ERROR: File not found: {path}")
                return False
    
    # Determine if this is an alternate evaluation
    use_alternate = config.get('use_alternate', False)
    
    try:
        gt_raw = load_json(config['ground_truth'])
        ret_raw = load_json(config['retrieval_results'])
        chunk_text_map = load_json(config['chunk_text'])
        
        gt_index = index_ground_truth(gt_raw)
        
        per_query_results = {}
        accum_doc = {k: [] for k in K_VALUES}
        accum_span_p = {k: [] for k in K_VALUES}
        accum_span_r = {k: [] for k in K_VALUES}
        accum_alt_p = {k: [] for k in K_VALUES}
        accum_alt_r = {k: [] for k in K_VALUES}
        
        # Additional accumulators for alternate approach
        if use_alternate:
            accum_drm = {k: [] for k in K_VALUES}
            # Case analysis buckets
            case_span_correct_alt_wrong = []
            case_span_wrong_alt_correct = []
            case_both_correct = []
            case_both_wrong = []
        
        processed_queries = 0
        
        iterator = tqdm(ret_raw.items()) if use_alternate else ret_raw.items()
        
        for qid, entry in iterator:
            q = entry["query"]
            if q not in gt_index:
                continue
            
            if use_alternate:
                metrics, span_flags, alt_flags = evaluate_query(
                    q,
                    gt_index[q],
                    entry["retrieved_chunks"],
                    chunk_text_map,
                    use_alternate=True
                )
                
                # Case analysis (top-1)
                span1 = span_flags[0] if span_flags else False
                alt1 = alt_flags[0] if alt_flags else False

                if span1 and not alt1:
                    case_span_correct_alt_wrong.append(qid)
                elif (not span1) and alt1:
                    case_span_wrong_alt_correct.append(qid)
                elif span1 and alt1:
                    case_both_correct.append(qid)
                else:
                    case_both_wrong.append(qid)
                
                # Accumulate DRM metrics
                for k in K_VALUES:
                    accum_drm[k].append(metrics["drm"][k])
            else:
                metrics = evaluate_query(
                    q,
                    gt_index[q],
                    entry["retrieved_chunks"],
                    chunk_text_map,
                    use_alternate=False
                )
            
            per_query_results[qid] = {"query": q, "metrics": metrics}
            
            for k in K_VALUES:
                accum_doc[k].append(metrics["doc_retrieved_at_k"][k])
                accum_span_p[k].append(metrics["span_precision"][k])
                accum_span_r[k].append(metrics["span_recall"][k])
                accum_alt_p[k].append(metrics["alt_precision"][k])
                accum_alt_r[k].append(metrics["alt_recall"][k])
            
            processed_queries += 1
        
        if processed_queries == 0:
            print(f"WARNING: No queries processed for {config['name']}")
            return False
        
        macro_results = {
            "doc_retrieved_macro": {k: sum(accum_doc[k]) / len(accum_doc[k]) for k in K_VALUES},
            "span_precision_macro": {k: sum(accum_span_p[k]) / len(accum_span_p[k]) for k in K_VALUES},
            "span_recall_macro": {k: sum(accum_span_r[k]) / len(accum_span_r[k]) for k in K_VALUES},
            "alt_precision_macro": {k: sum(accum_alt_p[k]) / len(accum_alt_p[k]) for k in K_VALUES},
            "alt_recall_macro": {k: sum(accum_alt_r[k]) / len(accum_alt_r[k]) for k in K_VALUES},
        }
        
        # Add DRM macro results for alternate approach
        if use_alternate:
            macro_results["drm_macro"] = {k: sum(accum_drm[k]) / len(accum_drm[k]) for k in K_VALUES}
        
        output = {
            "evaluation_name": config['name'],
            "per_query": per_query_results,
            "macro": macro_results,
            "k_values": K_VALUES,
            "total_queries_processed": processed_queries
        }
        
        # Add case analysis for alternate approach
        if use_alternate:
            total_queries = processed_queries
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
            output["cases"] = case_stats
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(config['output']), exist_ok=True)
        
        Path(config['output']).write_text(json.dumps(output, indent=2))
        print(f"✓ Saved results to {config['output']}")
        print(f"✓ Processed {processed_queries} queries")
        
        # Print summary metrics
        print(f"Summary for {config['name']}:")
        for k in [1, 4, 16]:
            doc_ret = macro_results["doc_retrieved_macro"][k]
            span_p = macro_results["span_precision_macro"][k]
            span_r = macro_results["span_recall_macro"][k]
            alt_p = macro_results["alt_precision_macro"][k]
            alt_r = macro_results["alt_recall_macro"][k]
            
            if use_alternate:
                drm = macro_results["drm_macro"][k]
                print(f"  k={k}: Doc={doc_ret:.3f}, Span P/R={span_p:.3f}/{span_r:.3f}, Alt P/R={alt_p:.3f}/{alt_r:.3f}, DRM={drm:.3f}")
            else:
                print(f"  k={k}: Doc={doc_ret:.3f}, Span P/R={span_p:.3f}/{span_r:.3f}, Alt P/R={alt_p:.3f}/{alt_r:.3f}")
        
        # Print case analysis for alternate approach
        if use_alternate:
            print(f"Case Analysis:")
            print(f"  Both Correct: {case_stats['both_correct']['count']} ({case_stats['both_correct']['percentage']:.1f}%)")
            print(f"  Span Correct, Alt Wrong: {case_stats['span_correct_alt_wrong']['count']} ({case_stats['span_correct_alt_wrong']['percentage']:.1f}%)")
            print(f"  Span Wrong, Alt Correct: {case_stats['span_wrong_alt_correct']['count']} ({case_stats['span_wrong_alt_correct']['percentage']:.1f}%)")
            print(f"  Both Wrong: {case_stats['both_wrong']['count']} ({case_stats['both_wrong']['percentage']:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"ERROR in {config['name']}: {str(e)}")
        return False

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Run all evaluations"""
    print("Starting comprehensive RAG evaluation...")
    print(f"Will run {len(EVALUATIONS)} evaluations")
    
    # Separate original and alternate evaluations
    original_evals = [e for e in EVALUATIONS if not e.get('use_alternate', False)]
    alternate_evals = [e for e in EVALUATIONS if e.get('use_alternate', False)]
    
    print(f"  - Original evaluations: {len(original_evals)}")
    print(f"  - Alternate evaluations (with DRM & case analysis): {len(alternate_evals)}")
    
    success_count = 0
    
    for config in EVALUATIONS:
        if run_single_evaluation(config):
            success_count += 1
    
    print(f"\n=== EVALUATION COMPLETE ===")
    print(f"Successfully completed: {success_count}/{len(EVALUATIONS)} evaluations")
    
    if success_count < len(EVALUATIONS):
        print("Some evaluations failed. Check the error messages above.")
    else:
        print("All evaluations completed successfully!")
        print("\nOutput files created:")
        for config in EVALUATIONS:
            eval_type = "ALTERNATE" if config.get('use_alternate', False) else "ORIGINAL"
            print(f"  [{eval_type}] {config['output']}")

if __name__ == "__main__":
    main()