import os
import json
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple


# =====================================================================
# Utility: Count tokens
# =====================================================================

def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# =====================================================================
# Forced token-based splitting (fallback)
# =====================================================================

def _force_token_based_splits(
    text: str,
    global_start: int,
    tokenizer,
    max_tokens: int,
    chunk_overlap: int
) -> List[Dict[str, Any]]:
    encoded = tokenizer(
        text, return_offsets_mapping=True, add_special_tokens=False
    )
    offsets = encoded["offset_mapping"]
    n_tokens = len(encoded["input_ids"])

    segments = []
    start_tok = 0

    while start_tok < n_tokens:
        end_tok = min(start_tok + max_tokens, n_tokens)
        local_char_start = offsets[start_tok][0]
        local_char_end = offsets[end_tok - 1][1]

        segment_text = text[local_char_start:local_char_end]

        segments.append({
            "text": segment_text,
            "start": global_start + local_char_start,
            "end": global_start + local_char_end,
            "n_tokens": end_tok - start_tok,
        })

        if end_tok == n_tokens:
            break

        start_tok += max_tokens - chunk_overlap

    return segments


# =====================================================================
# Recursive splitting using separators
# =====================================================================

def _recursive_split(
    text: str,
    global_start: int,
    tokenizer,
    max_tokens: int,
    separators: List[str],
    sep_idx: int,
    forced_overlap: int
) -> List[Dict[str, Any]]:

    n_tokens = _count_tokens(text, tokenizer)

    # BASE CASE: Fits in max token limit
    if n_tokens <= max_tokens:
        return [{
            "text": text,
            "start": global_start,
            "end": global_start + len(text),
            "n_tokens": n_tokens
        }]

    # If no separators left → force token splitting
    if sep_idx >= len(separators):
        return _force_token_based_splits(
            text,
            global_start,
            tokenizer,
            max_tokens,
            forced_overlap
        )

    sep = separators[sep_idx]

    # If separator not found, move to next separator
    if sep not in text:
        return _recursive_split(
            text,
            global_start,
            tokenizer,
            max_tokens,
            separators,
            sep_idx + 1,
            forced_overlap
        )

    segments = []
    pieces = []
    cursor = 0
    sep_len = len(sep)
    text_len = len(text)

    # Split preserving separator
    while cursor < text_len:
        idx = text.find(sep, cursor)
        if idx == -1:
            if cursor < text_len:
                pieces.append((cursor, text_len))
            break
        else:
            end_idx = idx + sep_len
            pieces.append((cursor, end_idx))
            cursor = end_idx

    # Recursively split each piece further
    for local_start, local_end in pieces:
        sub_text = text[local_start:local_end]
        sub_global_start = global_start + local_start

        sub_segments = _recursive_split(
            sub_text,
            sub_global_start,
            tokenizer,
            max_tokens,
            separators,
            sep_idx + 1,
            forced_overlap
        )
        segments.extend(sub_segments)

    return segments


# =====================================================================
# Merge segments into final chunks with overlap
# =====================================================================

def recursive_chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[Dict[str, Any]]:

    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    # Stage 1: recursively produce small segments
    segments = _recursive_split(
        text=text,
        global_start=0,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        separators=separators,
        sep_idx=0,
        forced_overlap=chunk_overlap
    )

    # Stage 2: merge segments into chunks with overlaps
    chunks = []
    current_segs = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = seg["n_tokens"]

        if current_segs and current_tokens + seg_tokens > max_tokens:
            chunk_text = "".join(s["text"] for s in current_segs)
            chunk_start = current_segs[0]["start"]
            chunk_end = current_segs[-1]["end"]

            chunks.append({
                "chunk_text": chunk_text,
                "span": [chunk_start, chunk_end]
            })

            # Build overlap window
            overlap_segs = []
            overlap_tok = 0

            for s in reversed(current_segs):
                if overlap_tok + s["n_tokens"] > chunk_overlap:
                    break
                overlap_segs.insert(0, s)
                overlap_tok += s["n_tokens"]

            current_segs = overlap_segs
            current_tokens = overlap_tok

        current_segs.append(seg)
        current_tokens += seg_tokens

    if current_segs:
        chunk_text = "".join(s["text"] for s in current_segs)
        chunk_start = current_segs[0]["start"]
        chunk_end = current_segs[-1]["end"]

        chunks.append({
            "chunk_text": chunk_text,
            "span": [chunk_start, chunk_end]
        })

    return chunks


# =====================================================================
# DOC-WISE CHUNKING WITH SAME ID FORMAT
# =====================================================================

def chunk_folder_llama_json_recursive_docwise(
    input_folder: str,
    output_folder: str,
    chunk_size: int = 500,
    window: int = 50,
    model_name: str = "thenlper/gte-large",
    separators: Optional[List[str]] = None
):
    """
    Produces ONE JSON PER DOCUMENT.
    Keys remain exactly the same:
        <folder>_<document>_chunk1
        <folder>_<document>_chunk2
    """
    os.makedirs(output_folder, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    folder_name = os.path.basename(os.path.normpath(input_folder))

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = recursive_chunk_text(
            text=text,
            tokenizer=tokenizer,
            max_tokens=chunk_size,
            chunk_overlap=window,
            separators=separators
        )

        doc_dict = {}
        for idx, ch in enumerate(chunks, start=1):
            chunk_id = f"{folder_name}_{base_name}_chunk{idx}"
            doc_dict[chunk_id] = ch

        # Sanitize filename for Windows (max 255 chars, remove problematic chars)
        safe_base_name = base_name.replace(".pdf__", "_").replace(":", "_")
        if len(safe_base_name) > 200:  # Leave room for .json extension
            safe_base_name = safe_base_name[:200]
        
        outpath = os.path.join(output_folder, f"{safe_base_name}.json")
        with open(outpath, "w", encoding="utf-8") as jf:
            json.dump(doc_dict, jf, ensure_ascii=False, indent=2)

        print(f"[✔] Saved {len(chunks)} chunks → {outpath}")


# =====================================================================
# Main execution
# =====================================================================

if __name__ == "__main__":
    chunk_folder_llama_json_recursive_docwise(
        input_folder="rag/data/privacy_qa",
        output_folder="RAG_pipeline/window_summary_chunking/chunks_only",
        chunk_size=380,
        window=50,
        model_name="thenlper/gte-large"
    )
    
    print("\n✓ All documents chunked successfully!")
