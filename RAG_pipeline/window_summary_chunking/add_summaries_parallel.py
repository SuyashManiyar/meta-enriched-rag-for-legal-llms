"""
Add metadata summaries using Groq API with parallel batch processing
Processes multiple windows concurrently while respecting rate limits
"""
import os
import json
import asyncio
from groq import AsyncGroq
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
WINDOW_SIZE = 4
SUMMARY_SIZE = 100
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_CONCURRENT_REQUESTS = 25  # Stay under 30 RPM limit
BATCH_SIZE = 10  # Process 10 files at a time

# Initialize
api_key = os.getenv("GROQ_API_KEY", "").strip()
client = AsyncGroq(api_key=api_key) if api_key else None


async def generate_summary_async(window_chunks: list, max_tokens: int = 100, max_retries: int = 3) -> str:
    """Generate summary using Groq async"""
    if not client:
        return "[Summary unavailable - no API key]"
    
    window_text = " ".join([c["chunk_text"] for c in window_chunks])
    
    prompt = f"""Summarize the following privacy policy text in {max_tokens} tokens or less.
Focus on key information: data collection, usage, sharing, and user rights.

Text:
---
{window_text}
---

Summary (max {max_tokens} tokens):"""

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a legal document summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_msg = str(e)
            if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    print(f"  [RETRY] Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
            
            print(f"  [ERROR] Summary failed: {e}")
            return f"[Summary generation failed: {error_msg[:100]}]"
    
    return "[Summary generation failed: Max retries exceeded]"


async def process_window(window, window_idx, semaphore):
    """Process a single window with rate limiting"""
    async with semaphore:
        summary = await generate_summary_async(window, max_tokens=SUMMARY_SIZE)
        return window_idx, summary


async def add_summaries_to_file_async(input_file: str, output_file: str):
    """Add summaries to a single chunk file using async processing"""
    filename = os.path.basename(input_file)
    print(f"\nProcessing: {filename}")
    
    # Load chunks
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Convert to list and sort
    chunk_list = []
    for chunk_id, chunk_data in chunks.items():
        chunk_list.append({
            "chunk_id": chunk_id,
            **chunk_data
        })
    
    chunk_list.sort(key=lambda x: int(x["chunk_id"].split("_chunk")[-1]))
    print(f"  Loaded {len(chunk_list)} chunks")
    
    # Create windows
    windows = []
    for i in range(0, len(chunk_list), WINDOW_SIZE):
        window = chunk_list[i:i + WINDOW_SIZE]
        windows.append((i, window))
    
    print(f"  Processing {len(windows)} windows in parallel...")
    
    # Process windows concurrently with rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [process_window(window, idx, semaphore) for idx, window in windows]
    
    # Gather results
    results = await asyncio.gather(*tasks)
    
    # Apply summaries to chunks
    for window_idx, summary in results:
        window_start = window_idx
        window = chunk_list[window_start:window_start + WINDOW_SIZE]
        
        for chunk in window:
            chunk["metadata_summary"] = summary
            chunk["chunk_text_with_metadata"] = f"{chunk['chunk_text']}\n\n<metadata>\n{summary}\n</metadata>"
            chunk["window_chunks"] = [c["chunk_id"] for c in window]
    
    # Convert back to dictionary
    result = {}
    for chunk in chunk_list:
        chunk_id = chunk.pop("chunk_id")
        result[chunk_id] = chunk
    
    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"  [OK] Saved to {output_file}")


async def process_batch_of_files(file_batch, input_folder, output_folder):
    """Process a batch of files concurrently"""
    tasks = []
    for json_file in file_batch:
        input_path = os.path.join(input_folder, json_file)
        base_name = json_file.replace('.json', '')
        output_filename = f"{base_name}_with_summaries.json"
        output_path = os.path.join(output_folder, output_filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {json_file} (already processed)")
            continue
        
        tasks.append(add_summaries_to_file_async(input_path, output_path))
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def process_all_chunk_files_async(input_folder: str, output_folder: str):
    """Process all chunk files in batches"""
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} chunk files")
    print(f"Processing in batches of {BATCH_SIZE} files\n")
    
    # Process files in batches
    for i in range(0, len(json_files), BATCH_SIZE):
        batch = json_files[i:i + BATCH_SIZE]
        print(f"\n=== Batch {i//BATCH_SIZE + 1}/{(len(json_files)-1)//BATCH_SIZE + 1} ===")
        
        try:
            await process_batch_of_files(batch, input_folder, output_folder)
        except Exception as e:
            print(f"[ERROR] Batch failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n[OK] All files processed!")


def main():
    """Main entry point"""
    import sys
    
    if not client:
        print("ERROR: GROQ_API_KEY not set!")
        print("\nGet your free API key from: https://console.groq.com/keys")
        exit(1)
    
    input_folder = "RAG_pipeline/window_summary_chunking/maud_chunks_only"
    output_folder = "RAG_pipeline/window_summary_chunking/maud_chunks_with_summaries"
    
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    print("Parallel Summary Generation with Groq")
    print("=" * 70)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Batch size: {BATCH_SIZE} files")
    print()
    
    # Run async processing
    start_time = time.time()
    asyncio.run(process_all_chunk_files_async(input_folder, output_folder))
    elapsed = time.time() - start_time
    
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
