import os
import json
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple


# ========================================================================================================================
# Fixed chunking implementation
# ========================================================================================================================

def chunk_folder_llama_json(
    input_folder: str,
    output_json: str,
    chunk_size: int = 380,
    window: int = 50,
    model_name: str = "thenlper/gte-large"
):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Final dictionary to save
    result = {}

    # Identify folder name for prefix
    folder_name = os.path.basename(os.path.normpath(input_folder))

    # Iterate over all .txt files
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(input_folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Remove extension for chunk_id
        base_name = os.path.splitext(filename)[0]

        # Tokenize with offsets
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        n = len(token_ids)

        start = 0
        chunk_idx = 1

        # Sliding window chunking
        while start < n:
            end = min(start + chunk_size, n)

            chunk_offsets = offsets[start:end]

            # Exact character span in original raw text
            char_start = chunk_offsets[0][0]
            char_end = chunk_offsets[-1][1]

            chunk_text = text[char_start:char_end]

            # Construct chunk ID (no .txt)
            chunk_id = f"{folder_name}_{base_name}_chunk{chunk_idx}"

            # Save into dictionary
            result[chunk_id] = {
                "chunk_text": chunk_text,
                "span": [char_start, char_end]
            }

            chunk_idx += 1
            start += chunk_size - window

    # Write entire dictionary as one JSON
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(result, jf, ensure_ascii=False, indent=2)

chunk_folder_llama_json(
    input_folder="/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud",
    output_json="/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_chunks/chunks_fixed.json",
    chunk_size=500,
    window=50,
    model_name="thenlper/gte-large"
)

# ========================================================================================================================
# Recursive chunking implementation
# ========================================================================================================================

def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _force_token_based_splits(
    text: str,
    global_start: int,
    tokenizer,
    max_tokens: int,
    chunk_overlap: int
) -> List[Dict[str, Any]]:
    """Fallback: strictly token-based splitting for a text segment that is
    still too long even after trying all separators.
    Returns list of segments with start/end and token counts.
    """
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
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
        n_seg_tokens = end_tok - start_tok

        segments.append({
            "text": segment_text,
            "start": global_start + local_char_start,
            "end": global_start + local_char_end,
            "n_tokens": n_seg_tokens,
        })

        if end_tok == n_tokens:
            break

        # Move by max_tokens - overlap in token space
        start_tok += max_tokens - chunk_overlap

    return segments


def _recursive_split(
    text: str,
    global_start: int,
    tokenizer,
    max_tokens: int,
    separators: List[str],
    sep_idx: int,
    forced_overlap: int
) -> List[Dict[str, Any]]:
    """Recursively split `text` into segments <= max_tokens tokens,
    using separators in `separators` from larger to smaller.
    """
    n_tokens = _count_tokens(text, tokenizer)

    # Base case: already small enough
    if n_tokens <= max_tokens:
        return [{
            "text": text,
            "start": global_start,
            "end": global_start + len(text),
            "n_tokens": n_tokens,
        }]

    # No more separators to try: force token-based splitting
    if sep_idx >= len(separators):
        return _force_token_based_splits(
            text,
            global_start=global_start,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            chunk_overlap=forced_overlap
        )

    sep = separators[sep_idx]

    # If current separator not found, try next
    if sep not in text:
        return _recursive_split(
            text=text,
            global_start=global_start,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            separators=separators,
            sep_idx=sep_idx + 1,
            forced_overlap=forced_overlap
        )

    # Split by current separator (keeping the separator at end of each piece)
    segments: List[Dict[str, Any]] = []
    pieces: List[Tuple[int, int]] = []  # local [start, end)

    cursor = 0
    sep_len = len(sep)
    text_len = len(text)

    while cursor < text_len:
        idx = text.find(sep, cursor)
        if idx == -1:
            # last piece
            if cursor < text_len:
                pieces.append((cursor, text_len))
            break
        else:
            end_idx = idx + sep_len
            pieces.append((cursor, end_idx))
            cursor = end_idx

    # Recursively process each piece with next separator
    for local_start, local_end in pieces:
        sub_text = text[local_start:local_end]
        sub_global_start = global_start + local_start
        sub_segments = _recursive_split(
            text=sub_text,
            global_start=sub_global_start,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            separators=separators,
            sep_idx=sep_idx + 1,
            forced_overlap=forced_overlap
        )
        segments.extend(sub_segments)

    return segments


def recursive_chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Recursive chunking:
    1) Recursively split text into small segments using separators
       ("\n\n", "\n", ". ", " "), forcing token splits if needed.
    2) Merge segments into final chunks with token-based overlap.
    Returns:
        List of {"chunk_text": ..., "span": [char_start, char_end]}
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    # Stage A: recursive segmentation into small segments
    segments = _recursive_split(
        text=text,
        global_start=0,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        separators=separators,
        sep_idx=0,
        forced_overlap=chunk_overlap
    )

    # Stage B: merge segments into overlapping chunks
    chunks: List[Dict[str, Any]] = []
    current_segs: List[Dict[str, Any]] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = seg["n_tokens"]

        # If adding this segment would overflow the chunk, finalize current chunk
        if current_segs and current_tokens + seg_tokens > max_tokens:
            chunk_text = "".join(s["text"] for s in current_segs)
            chunk_start = current_segs[0]["start"]
            chunk_end = current_segs[-1]["end"]

            chunks.append({
                "chunk_text": chunk_text,
                "span": [chunk_start, chunk_end],
            })

            # Build overlap tail from the end of current_segs
            overlap_segs: List[Dict[str, Any]] = []
            overlap_tokens = 0
            for s in reversed(current_segs):
                if overlap_tokens + s["n_tokens"] > chunk_overlap:
                    break
                overlap_segs.insert(0, s)  # prepend
                overlap_tokens += s["n_tokens"]

            current_segs = overlap_segs
            current_tokens = overlap_tokens

        # Add current segment
        current_segs.append(seg)
        current_tokens += seg_tokens

    # Flush any remaining segments as last chunk
    if current_segs:
        chunk_text = "".join(s["text"] for s in current_segs)
        chunk_start = current_segs[0]["start"]
        chunk_end = current_segs[-1]["end"]

        chunks.append({
            "chunk_text": chunk_text,
            "span": [chunk_start, chunk_end],
        })

    return chunks


def chunk_folder_llama_json_recursive(
    input_folder: str,
    output_json: str,
    chunk_size: int = 500,
    window: int = 50,
    model_name: str = "thenlper/gte-large",
    separators: Optional[List[str]] = None
):
    """Folder-level wrapper that uses recursive_chunk_text instead of
    fixed sliding-window chunking.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    result = {}
    folder_name = os.path.basename(os.path.normpath(input_folder))

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(input_folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        base_name = os.path.splitext(filename)[0]

        # Use recursive chunking
        chunks = recursive_chunk_text(
            text=text,
            tokenizer=tokenizer,
            max_tokens=chunk_size,
            chunk_overlap=window,
            separators=separators
        )

        # Add chunks to result with IDs
        for idx, ch in enumerate(chunks, start=1):
            chunk_id = f"{folder_name}_{base_name}_chunk{idx}"
            result[chunk_id] = ch

    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(result, jf, ensure_ascii=False, indent=2)

# chunk_folder_llama_json_recursive(
#     input_folder="/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa",
#     output_json="/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_recur_data/privacy_qa_chunks_recur.json",
#     chunk_size=500,
#     window=50,
#     model_name="thenlper/gte-large"
# )

