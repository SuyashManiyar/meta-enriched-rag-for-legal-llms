"""
Add metadata summaries using Groq API with batched windows
Generates multiple window summaries in one API call to reduce total API calls
"""
import os
import json
from groq import Groq
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
WINDOW_SIZE = 4
SUMMARY_SIZE = 100
GROQ_MODEL = "llama-3.3-70b-versatile"
WINDOWS_PER_BATCH = 2  # Generate 2 window summaries per API call

# Initialize
api_key = os.getenv("GROQ_API_KEY", "").strip()
client = Groq(api_key=api_key) if api_key else None


def generate_batched_summaries(windows_batch: list, max_tokens: int = 100, max_retries: int = 3) -> list:
    """
    Generate summaries for multiple windows in one API call
    
    Args:
        windows_batch: List of windows, each window is a list of chunks
        max_tokens: Max tokens per summary
        max_retries: Number of retries
    
    Returns:
        List of summaries, one per window
    """
    if not client:
        return ["[Summary unavailable - no API key]"] * len(windows_batch)
    
    # Build prompt for multiple windows with JSON output
    prompt_parts = [
        f"Summarize each of the {len(windows_batch)} text sections below.",
        f"Each summary should be {max_tokens} tokens or less.",
        "Focus on key information: data collection, usage, sharing, and user rights.",
        "",
        "Return ONLY a JSON array with summaries in order:",
        '["summary for section 1", "summary for section 2"]',
        ""
    ]
    
    for idx, window in enumerate(windows_batch, 1):
        window_text = " ".join([c["chunk_text"] for c in window])
        prompt_parts.append(f"=== SECTION {idx} ===")
        prompt_parts.append(window_text)
        prompt_parts.append("")
    
    prompt = "\n".join(prompt_parts)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a legal document summarizer. Return summaries as a JSON array."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens * len(windows_batch) + 100,  # Extra tokens for JSON formatting
                temperature=0
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                
                summaries = json.loads(response_text)
                
                # Ensure we have the right number of summaries
                if len(summaries) == len(windows_batch):
                    return summaries
                else:
                    print(f"  [WARNING] Expected {len(windows_batch)} summaries, got {len(summaries)}")
                    # Pad or truncate
                    while len(summaries) < len(windows_batch):
                        summaries.append("[Summary missing]")
                    return summaries[:len(windows_batch)]
                    
            except json.JSONDecodeError:
                # Fallback: split by newlines
                print(f"  [WARNING] JSON parsing failed, using fallback")
                lines = [l.strip() for l in response_text.split('\n') if l.strip() and not l.startswith('[') and not l.startswith(']')]
                summaries = []
                for line in lines:
                    # Remove quotes and commas
                    clean = line.strip('",').strip()
                    if clean:
                        summaries.append(clean)
                
                # Ensure correct length
                while len(summaries) < len(windows_batch):
                    summaries.append("[Summary extraction failed]")
                
                return summaries[:len(windows_batch)]
        
        except Exception as e:
            error_msg = str(e)
            if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    print(f"  [RETRY] Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            print(f"  [ERROR] Batch summary failed: {e}")
            return [f"[Summary generation failed: {error_msg[:100]}]"] * len(windows_batch)
    
    return ["[Summary generation failed: Max retries exceeded]"] * len(windows_batch)


def add_summaries_to_file(input_file: str, output_file: str):
    """Add summaries to a single chunk file with batched API calls"""
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
    
    total_api_calls = (len(windows) + WINDOWS_PER_BATCH - 1) // WINDOWS_PER_BATCH
    print(f"  Processing {len(windows)} windows in {total_api_calls} API calls (batch size: {WINDOWS_PER_BATCH})")
    
    # Process windows in batches
    for batch_idx in range(0, len(windows), WINDOWS_PER_BATCH):
        batch_windows = windows[batch_idx:batch_idx + WINDOWS_PER_BATCH]
        window_indices = [w[0] for w in batch_windows]
        window_data = [w[1] for w in batch_windows]
        
        print(f"  API call {batch_idx//WINDOWS_PER_BATCH + 1}/{total_api_calls}: Windows {min(window_indices)//WINDOW_SIZE + 1}-{max(window_indices)//WINDOW_SIZE + 1}...")
        
        # Generate summaries for this batch
        summaries = generate_batched_summaries(window_data, max_tokens=SUMMARY_SIZE)
        
        # Apply summaries to chunks
        for (window_idx, window), summary in zip(batch_windows, summaries):
            for chunk in window:
                chunk["metadata_summary"] = summary
                chunk["chunk_text_with_metadata"] = f"{chunk['chunk_text']}\n\n<metadata>\n{summary}\n</metadata>"
                chunk["window_chunks"] = [c["chunk_id"] for c in window]
        
        # INCREMENTAL SAVE after each batch
        result = {}
        for chunk in chunk_list:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                result[chunk_id] = {k: v for k, v in chunk.items() if k != "chunk_id"}
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Rate limit: Sleep 2 seconds between batches
        if batch_idx + WINDOWS_PER_BATCH < len(windows):
            time.sleep(2)
    
    print(f"  [OK] Saved to {output_file}")


def process_all_chunk_files(input_folder: str, output_folder: str):
    """Process all chunk files"""
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} chunk files\n")
    
    for json_file in json_files:
        input_path = os.path.join(input_folder, json_file)
        base_name = json_file.replace('.json', '')
        output_filename = f"{base_name}_with_summaries.json"
        output_path = os.path.join(output_folder, output_filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {json_file} (already processed)")
            continue
        
        try:
            add_summaries_to_file(input_path, output_path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print("\n[OK] All files processed!")


if __name__ == "__main__":
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
    
    print("Batched Window Summary Generation")
    print("=" * 70)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Windows per batch: {WINDOWS_PER_BATCH}")
    print()
    
    start_time = time.time()
    process_all_chunk_files(input_folder, output_folder)
    elapsed = time.time() - start_time
    
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
