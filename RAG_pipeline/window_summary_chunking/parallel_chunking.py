"""
Multi-threaded chunking with multiple OpenAI API keys
Each thread processes one document with its own API key
"""
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from openai import OpenAI
import threading

# Configuration
CHUNK_SIZE = 380
WINDOW_SIZE = 4
SUMMARY_SIZE = 100
MAX_TOTAL_TOKENS = 500
MODEL_NAME = "thenlper/gte-large"
OPENAI_MODEL = "gpt-4o-mini"

# Load tokenizer once (thread-safe)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Multiple API keys (add your keys here)
API_KEYS = [
    os.getenv("OPENAI_API_KEY_1", ""),
    os.getenv("OPENAI_API_KEY_2", ""),
    os.getenv("OPENAI_API_KEY_3", ""),
    # Add more keys as needed
]

# Remove empty keys
API_KEYS = [k.strip() for k in API_KEYS if k.strip()]


def generate_window_summary(client, window_text: str, max_tokens: int = 100, max_retries: int = 3) -> str:
    """Generate summary using provided client"""
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
                timeout=60
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    print(f"  ⚠ Retry in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            print(f"  ✗ Summary failed: {e}")
            return f"[Summary generation failed: {error_msg[:100]}]"
    
    return "[Summary generation failed: Max retries exceeded]"


def recursive_split(text: str, max_tokens: int = CHUNK_SIZE):
    """Recursively split text"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    # Try paragraphs
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        chunks = []
        for para in paragraphs:
            chunks.extend(recursive_split(para, max_tokens))
        return chunks
    
    # Try lines
    lines = text.split('\n')
    if len(lines) > 1:
        chunks = []
        for line in lines:
            chunks.extend(recursive_split(line, max_tokens))
        return chunks
    
    # Try sentences
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


def process_document_with_key(input_path: str, output_path: str, api_key: str, thread_id: int):
    """Process one document with a specific API key"""
    filename = os.path.basename(input_path)
    thread_name = threading.current_thread().name
    
    print(f"[Thread-{thread_id}] Starting: {filename}")
    
    # Create client for this thread
    client = OpenAI(api_key=api_key)
    
    try:
        # Read document
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Chunk
        print(f"[Thread-{thread_id}] Chunking {filename}...")
        text_chunks = recursive_split(text, max_tokens=CHUNK_SIZE)
        print(f"[Thread-{thread_id}] Created {len(text_chunks)} chunks")
        
        # Process windows
        folder_name = "privacy_qa"
        base_name = filename.replace(".txt", "")
        result = {}
        window_chunks = []
        char_offset = 0
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            # Find positions
            char_start = text.find(chunk_text, char_offset)
            if char_start == -1:
                char_start = char_offset
            char_end = char_start + len(chunk_text)
            char_offset = char_end
            
            chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
            
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
            if len(window_chunks) == WINDOW_SIZE or chunk_idx == len(text_chunks) - 1:
                print(f"[Thread-{thread_id}] Summary for chunks {chunk_idx - len(window_chunks) + 1}-{chunk_idx}")
                
                window_text = " ".join([c["text"] for c in window_chunks])
                summary = generate_window_summary(client, window_text, max_tokens=SUMMARY_SIZE)
                
                for chunk in window_chunks:
                    chunk["text_with_metadata"] = f"{chunk['text']}\n\n<metadata>\n{summary}\n</metadata>"
                    chunk["metadata_summary"] = summary
                    
                    total_tokens = tokenizer.encode(chunk["text_with_metadata"], add_special_tokens=False)
                    chunk["total_token_count"] = len(total_tokens)
                    
                    if len(total_tokens) > MAX_TOTAL_TOKENS:
                        print(f"[Thread-{thread_id}] ⚠ Chunk {chunk['chunk_id']} = {len(total_tokens)} tokens")
                    
                    chunk_id = f"{folder_name}_{base_name}_chunk{chunk['chunk_id'] + 1}"
                    result[chunk_id] = {
                        "chunk_text": chunk["text_with_metadata"],
                        "span": [chunk["char_start"], chunk["char_end"]],
                        "metadata_summary": chunk["metadata_summary"],
                        "original_chunk_text": chunk["text"],
                        "original_token_count": chunk["token_count"],
                        "total_token_count": chunk["total_token_count"]
                    }
                
                # Save incrementally
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                window_chunks = []
        
        print(f"[Thread-{thread_id}] ✓ Completed {filename} → {len(result)} chunks")
        return {"success": True, "filename": filename, "chunks": len(result)}
    
    except Exception as e:
        print(f"[Thread-{thread_id}] ✗ ERROR in {filename}: {e}")
        return {"success": False, "filename": filename, "error": str(e)}


def process_all_parallel(input_folder: str, output_folder: str, max_workers: int = None):
    """Process all documents in parallel with different API keys"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Get files to process
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    # Filter already processed
    files_to_process = []
    for filename in txt_files:
        output_path = os.path.join(output_folder, filename.replace('.txt', '_windowed_chunks.json'))
        if not os.path.exists(output_path):
            files_to_process.append(filename)
    
    if not files_to_process:
        print("All files already processed!")
        return
    
    print(f"Processing {len(files_to_process)} documents in parallel")
    print(f"Using {len(API_KEYS)} API keys\n")
    
    # Limit workers to number of API keys
    if max_workers is None:
        max_workers = min(len(API_KEYS), len(files_to_process))
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i, filename in enumerate(files_to_process):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.txt', '_windowed_chunks.json'))
            api_key = API_KEYS[i % len(API_KEYS)]  # Rotate through API keys
            
            future = executor.submit(process_document_with_key, input_path, output_path, api_key, i)
            futures.append(future)
        
        # Wait for completion
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Summary
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    successful = sum(1 for r in results if r["success"])
    print(f"Successful: {successful}/{len(results)}")
    for r in results:
        if r["success"]:
            print(f"  ✓ {r['filename']}: {r['chunks']} chunks")
        else:
            print(f"  ✗ {r['filename']}: {r['error']}")


if __name__ == "__main__":
    if not API_KEYS:
        print("ERROR: No API keys found!")
        print("Set them with:")
        print("  $env:OPENAI_API_KEY_1 = 'key1'")
        print("  $env:OPENAI_API_KEY_2 = 'key2'")
        print("  $env:OPENAI_API_KEY_3 = 'key3'")
        exit(1)
    
    input_folder = "rag/data/privacy_qa"
    output_folder = "RAG_pipeline/window_summary_chunking/output"
    
    process_all_parallel(input_folder, output_folder)
    
    print("\n✓ All documents processed!")
