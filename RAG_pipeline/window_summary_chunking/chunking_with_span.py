"""
Window-based chunking with metadata summaries
- Chunk size: 400 tokens
- Window size: 4 chunks (1600 tokens)
- Generate 100-token summary for each window
- Append summary to chunks with <metadata> tag
"""
import os
import json
from transformers import AutoTokenizer
from openai import OpenAI

# Configuration
CHUNK_SIZE = 380  # tokens per chunk (leave room for metadata)
WINDOW_SIZE = 4   # chunks per window
SUMMARY_SIZE = 100  # tokens for summary
OVERLAP = 50  # token overlap between chunks
MAX_TOTAL_TOKENS = 500  # Maximum tokens including metadata

MODEL_NAME = "thenlper/gte-large"
OPENAI_MODEL = "gpt-4o-mini"

# Initialize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=api_key) if api_key else None


def generate_window_summary(window_text: str, max_tokens: int = 100, max_retries: int = 3) -> str:
    """Generate a concise summary of a text window using OpenAI with retry logic"""
    if not client:
        return "[Summary unavailable - no API key]"
    
    prompt = f"""Summarize the following privacy policy text in {max_tokens} tokens or less.
Focus on key information: data collection, usage, sharing, and user rights.

Text:
---
{window_text}
---

Summary (max {max_tokens} tokens):"""

    import time
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a legal document summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0,
                timeout=60  # 60 second timeout
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a timeout or rate limit error
            if "timeout" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                    print(f"  ⚠ Timeout/Rate limit (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            # For other errors or final attempt, return error message
            print(f"  ✗ Summary generation failed after {attempt + 1} attempts: {e}")
            return f"[Summary generation failed: {error_msg[:100]}]"
    
    return "[Summary generation failed: Max retries exceeded]"


def recursive_split(text: str, max_tokens: int = CHUNK_SIZE):
    """
    Recursively split text by paragraphs, then sentences, then tokens
    """
    # Count tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    # Try splitting by double newline (paragraphs)
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        chunks = []
        for para in paragraphs:
            chunks.extend(recursive_split(para, max_tokens))
        return chunks
    
    # Try splitting by single newline
    lines = text.split('\n')
    if len(lines) > 1:
        chunks = []
        for line in lines:
            chunks.extend(recursive_split(line, max_tokens))
        return chunks
    
    # Try splitting by sentences (period + space)
    sentences = text.split('. ')
    if len(sentences) > 1:
        chunks = []
        current = ""
        for sent in sentences:
            test = current + sent + ". "
            if len(tokenizer.encode(test, add_special_tokens=False)) <= max_tokens:
                current = test
            else:
                if current:
                    chunks.append(current.strip())
                current = sent + ". "
        if current:
            chunks.append(current.strip())
        return chunks
    
    # Fallback: split by tokens
    chunk_text = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    remaining_text = tokenizer.decode(tokens[max_tokens:], skip_special_tokens=True)
    return [chunk_text] + recursive_split(remaining_text, max_tokens)


def chunk_with_window_summaries(
    text: str,
    filename: str,
    output_path: str,
    chunk_size: int = CHUNK_SIZE,
    window_size: int = WINDOW_SIZE
):
    """
    Recursively chunk text and add metadata summaries every N chunks
    Saves incrementally after each window to avoid data loss
    
    Returns: List of chunks with metadata
    """
    # Recursive chunking
    print(f"  Performing recursive chunking (max {chunk_size} tokens)...")
    text_chunks = recursive_split(text, max_tokens=chunk_size)
    print(f"  Created {len(text_chunks)} chunks")
    
    chunks = []
    window_chunks = []
    char_offset = 0
    
    # Prepare output structure
    folder_name = "privacy_qa"
    base_name = filename.replace(".txt", "")
    result = {}
    
    for chunk_idx, chunk_text in enumerate(text_chunks):
        # Find character positions
        char_start = text.find(chunk_text, char_offset)
        if char_start == -1:
            char_start = char_offset
        char_end = char_start + len(chunk_text)
        char_offset = char_end
        
        # Count tokens
        chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
        
        # Store chunk info
        chunk_info = {
            "chunk_id": chunk_idx,
            "text": chunk_text,
            "char_start": char_start,
            "char_end": char_end,
            "token_count": len(chunk_tokens),
            "source": filename
        }
        
        window_chunks.append(chunk_info)
        
        # Generate summary every WINDOW_SIZE chunks
        if len(window_chunks) == window_size or chunk_idx == len(text_chunks) - 1:
            print(f"  Generating summary for chunks {chunk_idx - len(window_chunks) + 1}-{chunk_idx}...")
            
            # Combine window text
            window_text = " ".join([c["text"] for c in window_chunks])
            
            # Generate summary
            summary = generate_window_summary(window_text, max_tokens=SUMMARY_SIZE)
            
            # Add summary to each chunk in the window
            for chunk in window_chunks:
                chunk["text_with_metadata"] = f"{chunk['text']}\n\n<metadata>\n{summary}\n</metadata>"
                chunk["metadata_summary"] = summary
                
                # Verify total token count
                total_tokens = tokenizer.encode(chunk["text_with_metadata"], add_special_tokens=False)
                chunk["total_token_count"] = len(total_tokens)
                
                # Warning if exceeds max tokens
                if len(total_tokens) > MAX_TOTAL_TOKENS:
                    print(f"  ⚠ Warning: Chunk {chunk['chunk_id']} with metadata = {len(total_tokens)} tokens (>{MAX_TOTAL_TOKENS})")
                
                chunks.append(chunk)
                
                # Add to result dictionary
                chunk_id = f"{folder_name}_{base_name}_chunk{chunk['chunk_id'] + 1}"
                result[chunk_id] = {
                    "chunk_text": chunk["text_with_metadata"],
                    "span": [chunk["char_start"], chunk["char_end"]],
                    "metadata_summary": chunk["metadata_summary"],
                    "original_chunk_text": chunk["text"],
                    "original_token_count": chunk["token_count"],
                    "total_token_count": chunk["total_token_count"]  # With metadata
                }
            
            # SAVE INCREMENTALLY after each window
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  [SAVED] {len(result)} chunks so far...")
            
            # Clear window
            window_chunks = []
    
    return chunks


def process_document(input_path: str, output_path: str):
    """Process a single document with window summaries (saves incrementally)"""
    filename = os.path.basename(input_path)
    print(f"\nProcessing: {filename}")
    
    # Read document
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Chunk with summaries (saves incrementally inside)
    chunks = chunk_with_window_summaries(text, filename, output_path)
    
    print(f"  ✓ Completed {len(chunks)} chunks → {output_path}")


def process_all_documents(input_folder: str, output_folder: str):
    """Process all documents in folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} documents\n")
    
    for i, filename in enumerate(txt_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace('.txt', '_windowed_chunks.json')
        output_path = os.path.join(output_folder, output_filename)
        
        # Skip if exists
        if os.path.exists(output_path):
            print(f"[{i}/{len(txt_files)}] Skipping {filename} (already processed)")
            continue
        
        print(f"[{i}/{len(txt_files)}]", end=" ")
        
        try:
            process_document(input_path, output_path)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Check API key
    if not client:
        print("ERROR: OPENAI_API_KEY not set!")
        print("Set it with: $env:OPENAI_API_KEY = 'your-key'")
        exit(1)
    
    # Process all documents or specific ones
    input_folder = "rag/data/privacy_qa"
    output_folder = "RAG_pipeline/window_summary_chunking/output"
    
    # Optional: Process specific files to avoid rate limits
    # Usage: python chunking_with_span.py file1.txt file2.txt
    if len(sys.argv) > 1:
        specific_files = sys.argv[1:]
        print(f"Processing specific files: {specific_files}\n")
        for filename in specific_files:
            input_path = f"{input_folder}/{filename}"
            output_path = f"{output_folder}/{filename.replace('.txt', '_windowed_chunks.json')}"
            try:
                process_document(input_path, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    else:
        process_all_documents(input_folder, output_folder)
    
    print("\n✓ Processing complete!")
