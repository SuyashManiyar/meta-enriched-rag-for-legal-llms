"""
Add metadata summaries using Groq API (faster, higher limits)
"""
import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
WINDOW_SIZE = 4
SUMMARY_SIZE = 100
GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast and good quality (updated model)

# Initialize
api_key = os.getenv("GROQ_API_KEY", "").strip()
client = Groq(api_key=api_key) if api_key else None


def generate_summary(window_chunks: list, max_tokens: int = 100, max_retries: int = 3) -> str:
    """Generate summary using Groq"""
    if not client:
        return "[Summary unavailable - no API key]"
    
    # Combine chunk texts
    window_text = " ".join([c["chunk_text"] for c in window_chunks])
    
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
                    time.sleep(wait_time)
                    continue
            
            print(f"  [ERROR] Summary failed: {e}")
            return f"[Summary generation failed: {error_msg[:100]}]"
    
    return "[Summary generation failed: Max retries exceeded]"


def add_summaries_to_file(input_file: str, output_file: str):
    """Add summaries to a single chunk file"""
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
    
    # Process in windows with incremental saving
    import time
    for i in range(0, len(chunk_list), WINDOW_SIZE):
        window = chunk_list[i:i + WINDOW_SIZE]
        window_nums = [int(c["chunk_id"].split("_chunk")[-1]) for c in window]
        
        print(f"  Generating summary for chunks {min(window_nums)}-{max(window_nums)}...")
        
        # Generate summary
        summary = generate_summary(window, max_tokens=SUMMARY_SIZE)
        
        # Add summary to each chunk
        for chunk in window:
            chunk["metadata_summary"] = summary
            chunk["chunk_text_with_metadata"] = f"{chunk['chunk_text']}\n\n<metadata>\n{summary}\n</metadata>"
            chunk["window_chunks"] = [c["chunk_id"] for c in window]
        
        # INCREMENTAL SAVE: Save after each window to avoid losing progress
        result = {}
        for chunk in chunk_list:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                result[chunk_id] = {k: v for k, v in chunk.items() if k != "chunk_id"}
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Rate limit: Sleep 1 second between API calls to stay under 30 RPM
        if i + WINDOW_SIZE < len(chunk_list):  # Don't sleep after last window
            time.sleep(1)
    
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
        print("Then set it with: $env:GROQ_API_KEY = 'your-key'")
        exit(1)
    
    input_folder = "RAG_pipeline/window_summary_chunking/maud_chunks_only"
    output_folder = "RAG_pipeline/window_summary_chunking/maud_chunks_with_summaries"
    
    if len(sys.argv) > 1:
        # Process specific files
        specific_files = sys.argv[1:]
        print(f"Processing specific files: {specific_files}\n")
        
        os.makedirs(output_folder, exist_ok=True)
        
        for filename in specific_files:
            if not filename.endswith('.json'):
                filename = f"{filename}.json"
            
            input_path = os.path.join(input_folder, filename)
            
            if not os.path.exists(input_path):
                print(f"[ERROR] File not found: {filename}")
                continue
            
            base_name = filename.replace('.json', '')
            output_filename = f"{base_name}_with_summaries.json"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                add_summaries_to_file(input_path, output_path)
            except Exception as e:
                print(f"  [ERROR] {e}")
        
        print("\n[OK] Processing complete!")
    else:
        # Process all files
        process_all_chunk_files(input_folder, output_folder)
